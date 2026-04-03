#!/usr/bin/env python3
"""
Comprehensive benchmark for nano-vllm inference engine.

Compares Autoregressive (AR) baseline against EAGLE-3 speculative decoding,
collecting throughput, latency, VRAM, and acceptance metrics.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sharegpt_prompts(
    data_path: str,
    tokenizer,
    num_samples: int,
    max_model_len: int,
    min_tokens: int = 10,
    seed: int = 42,
) -> list[str]:
    """Load and filter ShareGPT prompts.

    Extracts the first human turn from randomly sampled conversations,
    filters out prompts that are too short (< min_tokens) or too long
    to leave room for generation within max_model_len.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract first human prompt from each conversation
    raw_prompts = []
    for conv in data:
        turns = conv.get("conversations", [])
        for turn in turns:
            if turn.get("from") == "human" and turn.get("value", "").strip():
                raw_prompts.append(turn["value"].strip())
                break

    rng = random.Random(seed)
    rng.shuffle(raw_prompts)

    # Apply chat template and filter by token length
    filtered: list[str] = []
    max_prompt_tokens = max_model_len // 2  # leave room for generation

    for text in raw_prompts:
        if len(filtered) >= num_samples * 3:
            break  # over-sample then pick

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        n_tok = len(tokenizer.encode(prompt))
        if n_tok < min_tokens or n_tok > max_prompt_tokens:
            continue
        filtered.append(prompt)

    if len(filtered) < num_samples:
        print(
            f"[WARN] Only {len(filtered)} valid prompts found "
            f"(requested {num_samples}). Using all available.",
            file=sys.stderr,
        )
    else:
        filtered = filtered[:num_samples]

    return filtered


# ---------------------------------------------------------------------------
# Per-request metrics
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    request_id: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0          # time to first token
    total_decode_ms: float = 0.0  # total decoding wall time
    tpt_ms: float = 0.0           # time per output token (avg)
    # EAGLE-3 specific (zero for AR)
    num_spec_steps: int = 0
    total_drafted: int = 0
    total_accepted: int = 0
    acceptance_counts: list[int] = field(default_factory=list)
    draft_head_time_ms: float = 0.0
    verify_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Benchmark engine wrapper
# ---------------------------------------------------------------------------

class BenchmarkEngine:
    """Thin wrapper around LLMEngine that instruments each step."""

    def __init__(self, engine: LLMEngine, mode: str, num_speculative_tokens: int):
        self.engine = engine
        self.mode = mode
        self.k = num_speculative_tokens

        # aggregate state
        self.total_output_tokens = 0
        self.per_request: dict[int, RequestMetrics] = {}
        self._active_requests: dict[int, dict] = {}  # seq_id -> tracking state

    # -- public API -----------------------------------------------------------

    @torch.inference_mode()
    def run(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
    ) -> list[RequestMetrics]:
        """Run full benchmark returning per-request metrics."""

        tokenizer = self.engine.tokenizer

        # Encode prompts
        token_id_lists = [tokenizer.encode(p) for p in prompts]

        for i, tids in enumerate(token_id_lists):
            sp = sampling_params
            self.engine.add_request(tids, sp)
            # map seq_id -> request bookkeeping
            # Sequence.counter is global; seq_id = latest counter - 1
            seq_id = Sequence.counter.__wrapped__ if hasattr(Sequence.counter, '__wrapped__') else None
            # We'll resolve seq_ids after first step by reading scheduler

        # Resolve seq_ids from scheduler
        all_seqs = list(self.engine.scheduler.waiting) + list(self.engine.scheduler.running)
        all_seqs.sort(key=lambda s: s.seq_id)

        for i, seq in enumerate(all_seqs):
            m = RequestMetrics(request_id=i, prompt_tokens=seq.num_prompt_tokens)
            self.per_request[seq.seq_id] = m
            self._active_requests[seq.seq_id] = {
                "prefilled": False,
                "decode_start_event": None,
                "first_token_event": None,
            }

        # VRAM tracking
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated()

        # Main loop with GPU timing
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)

        total_start.record()

        # aggregate EAGLE-3 counters
        agg_spec_steps = 0
        agg_drafted = 0
        agg_accepted = 0
        agg_acceptance_hist: list[int] = []  # per-step accepted counts
        agg_draft_head_ms = 0.0
        agg_verify_ms = 0.0

        # phase timing accumulators
        total_prefill_ms = 0.0
        total_decode_ms = 0.0
        num_decode_steps = 0
        first_decode_done = False

        while not self.engine.is_finished():
            if self.engine.use_speculative:
                seqs, is_prefill = self.engine.scheduler.schedule_speculative(self.k)
            else:
                seqs, is_prefill = self.engine.scheduler.schedule()

            if is_prefill:
                prefill_start.record()

                if self.engine.use_speculative:
                    token_ids = self.engine.model_runner.call(
                        "run_prefill_with_capture", seqs
                    )
                else:
                    token_ids = self.engine.model_runner.call("run", seqs, True)

                prefill_end.record()
                torch.cuda.synchronize()
                step_ms = prefill_start.elapsed_time(prefill_end)
                total_prefill_ms += step_ms

                self.engine.scheduler.postprocess(seqs, token_ids)

                # Record TTFT for each prefilled sequence
                for seq in seqs:
                    if seq.seq_id in self.per_request:
                        self.per_request[seq.seq_id].ttft_ms = step_ms
                        self._active_requests[seq.seq_id]["prefilled"] = True

                # Handle finished
                for seq in seqs:
                    if seq.is_finished and seq.seq_id in self.per_request:
                        self.per_request[seq.seq_id].completion_tokens = seq.num_completion_tokens

            else:
                # Decode step
                decode_start.record()

                if self.engine.use_speculative:
                    block_manager = self.engine.scheduler.block_manager

                    # --- Draft head timing ---
                    draft_ev_start = torch.cuda.Event(enable_timing=True)
                    draft_ev_end = torch.cuda.Event(enable_timing=True)
                    draft_ev_start.record()

                    k = self.k
                    mr = self.engine.model_runner
                    all_draft_tokens = []
                    for seq in seqs:
                        fused = mr.draft_model.fuse_features(
                            mr.saved_hidden[seq.seq_id]
                        )
                        drafts = mr.draft_model.generate(
                            mr.model.model.embed_tokens,
                            fused,
                            start_token=torch.tensor(
                                [seq.last_token], device="cuda", dtype=torch.long
                            ),
                            start_pos=len(seq),
                            k=k,
                        )
                        all_draft_tokens.append(drafts)

                    draft_ev_end.record()

                    # --- Pre-allocate blocks ---
                    for seq, drafts in zip(seqs, all_draft_tokens):
                        block_manager.pre_allocate_speculative(seq, len(drafts) + 1)

                    # --- Verification timing ---
                    verify_ev_start = torch.cuda.Event(enable_timing=True)
                    verify_ev_end = torch.cuda.Event(enable_timing=True)
                    verify_ev_start.record()

                    input_ids, positions = mr.prepare_verify(seqs, all_draft_tokens)
                    hidden_states, captured = mr.model(
                        input_ids, positions, capture_layers=mr.capture_layers
                    )
                    logits = mr.model.compute_logits_all(hidden_states)

                    from nanovllm.utils.context import reset_context
                    reset_context()

                    verify_ev_end.record()

                    # --- Acceptance ---
                    results = []
                    offset = 0
                    for i, seq in enumerate(seqs):
                        n_tokens = len(all_draft_tokens[i]) + 1
                        seq_logits = logits[offset:offset + n_tokens]
                        seq_captured = {
                            l: captured[l][offset:offset + n_tokens]
                            for l in captured
                        }
                        accepted, new_token, new_hidden = mr.accept_tokens(
                            seq_logits, all_draft_tokens[i], seq_captured
                        )
                        results.append((accepted, new_token))
                        mr.saved_hidden[seq.seq_id] = new_hidden
                        offset += n_tokens

                    decode_end.record()
                    torch.cuda.synchronize()

                    step_draft_ms = draft_ev_start.elapsed_time(draft_ev_end)
                    step_verify_ms = verify_ev_start.elapsed_time(verify_ev_end)
                    step_ms = decode_start.elapsed_time(decode_end)

                    # Accumulate EAGLE-3 metrics
                    for seq_idx, (accepted, new_token) in enumerate(results):
                        seq = seqs[seq_idx]
                        n_accepted = len(accepted)
                        n_drafted = len(all_draft_tokens[seq_idx])

                        agg_spec_steps += 1
                        agg_drafted += n_drafted
                        agg_accepted += n_accepted
                        agg_acceptance_hist.append(n_accepted)
                        agg_draft_head_ms += step_draft_ms
                        agg_verify_ms += step_verify_ms

                        if seq.seq_id in self.per_request:
                            m = self.per_request[seq.seq_id]
                            m.num_spec_steps += 1
                            m.total_drafted += n_drafted
                            m.total_accepted += n_accepted
                            m.acceptance_counts.append(n_accepted)
                            m.draft_head_time_ms += step_draft_ms
                            m.verify_time_ms += step_verify_ms

                    # Postprocess via scheduler
                    self.engine.scheduler.postprocess_speculative(seqs, results)

                    for seq in seqs:
                        if seq.is_finished and seq.seq_id in self.per_request:
                            self.per_request[seq.seq_id].completion_tokens = (
                                seq.num_completion_tokens
                            )
                            self.engine.model_runner.call(
                                "clear_saved_hidden", seq.seq_id
                            )

                else:
                    # AR decode
                    token_ids = self.engine.model_runner.call("run", seqs, False)
                    decode_end.record()
                    torch.cuda.synchronize()
                    step_ms = decode_start.elapsed_time(decode_end)

                    self.engine.scheduler.postprocess(seqs, token_ids)

                    for seq in seqs:
                        if seq.is_finished and seq.seq_id in self.per_request:
                            self.per_request[seq.seq_id].completion_tokens = (
                                seq.num_completion_tokens
                            )

                total_decode_ms += step_ms
                num_decode_steps += 1

        total_end.record()
        torch.cuda.synchronize()
        total_wall_ms = total_start.elapsed_time(total_end)

        vram_peak = torch.cuda.max_memory_allocated()

        # Compute per-request TPT
        total_completion = 0
        for m in self.per_request.values():
            total_completion += m.completion_tokens

        for m in self.per_request.values():
            if m.completion_tokens > 0 and total_decode_ms > 0:
                # Approximate per-request decode time proportionally
                share = m.completion_tokens / max(total_completion, 1)
                m.total_decode_ms = total_decode_ms * share
                m.tpt_ms = m.total_decode_ms / m.completion_tokens

        # Build summary
        summary = BenchmarkSummary(
            mode=self.mode,
            num_requests=len(self.per_request),
            total_prompt_tokens=sum(m.prompt_tokens for m in self.per_request.values()),
            total_completion_tokens=total_completion,
            total_wall_ms=total_wall_ms,
            total_prefill_ms=total_prefill_ms,
            total_decode_ms=total_decode_ms,
            num_decode_steps=num_decode_steps,
            vram_peak_bytes=vram_peak,
            # EAGLE-3
            spec_steps=agg_spec_steps,
            total_drafted=agg_drafted,
            total_accepted=agg_accepted,
            acceptance_hist=agg_acceptance_hist,
            draft_head_ms=agg_draft_head_ms,
            verify_ms=agg_verify_ms,
        )
        self._summary = summary

        return list(self.per_request.values())

    @property
    def summary(self):
        return self._summary


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSummary:
    mode: str
    num_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_wall_ms: float
    total_prefill_ms: float
    total_decode_ms: float
    num_decode_steps: int
    vram_peak_bytes: int
    # EAGLE-3
    spec_steps: int = 0
    total_drafted: int = 0
    total_accepted: int = 0
    acceptance_hist: list[int] = field(default_factory=list)
    draft_head_ms: float = 0.0
    verify_ms: float = 0.0

    @property
    def throughput_tps(self) -> float:
        if self.total_wall_ms <= 0:
            return 0.0
        return self.total_completion_tokens / (self.total_wall_ms / 1000.0)

    @property
    def avg_ttft_ms(self) -> float:
        return self.total_prefill_ms  # single-batch prefill approximation

    @property
    def avg_tpt_ms(self) -> float:
        if self.total_completion_tokens <= 0:
            return 0.0
        return self.total_decode_ms / self.total_completion_tokens

    @property
    def vram_peak_mb(self) -> float:
        return self.vram_peak_bytes / (1024 * 1024)

    @property
    def mean_acceptance_length(self) -> float:
        if not self.acceptance_hist:
            return 0.0
        return sum(self.acceptance_hist) / len(self.acceptance_hist)

    @property
    def avg_draft_head_ms(self) -> float:
        if self.spec_steps <= 0:
            return 0.0
        return self.draft_head_ms / self.spec_steps

    @property
    def avg_verify_ms(self) -> float:
        if self.spec_steps <= 0:
            return 0.0
        return self.verify_ms / self.spec_steps

    @property
    def tree_efficiency(self) -> float:
        if self.total_drafted <= 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    def acceptance_distribution(self, max_k: int = 10) -> dict[int, int]:
        dist: dict[int, int] = {i: 0 for i in range(max_k + 1)}
        for count in self.acceptance_hist:
            bucket = min(count, max_k)
            dist[bucket] += 1
        return dist


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_summary(summary: BenchmarkSummary, k: int):
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  BENCHMARK RESULTS  [{summary.mode.upper()}]")
    print(sep)

    rows = [
        ("Requests", f"{summary.num_requests}"),
        ("Total prompt tokens", f"{summary.total_prompt_tokens}"),
        ("Total completion tokens", f"{summary.total_completion_tokens}"),
        ("Wall time", f"{summary.total_wall_ms / 1000:.2f} s"),
        ("Throughput (TPS)", f"{summary.throughput_tps:.2f} tok/s"),
        ("Avg TTFT", f"{summary.avg_ttft_ms:.2f} ms"),
        ("Avg TPT", f"{summary.avg_tpt_ms:.2f} ms"),
        ("VRAM peak", f"{summary.vram_peak_mb:.1f} MB"),
    ]

    if summary.mode == "eagle-3":
        rows += [
            ("", ""),
            ("--- EAGLE-3 Metrics ---", ""),
            ("Speculative steps", f"{summary.spec_steps}"),
            (f"Mean acceptance length (tau)", f"{summary.mean_acceptance_length:.2f}"),
            ("Tree efficiency", f"{summary.tree_efficiency:.2%}"),
            ("Avg draft head time", f"{summary.avg_draft_head_ms:.2f} ms"),
            ("Avg verification time", f"{summary.avg_verify_ms:.2f} ms"),
        ]
        dist = summary.acceptance_distribution(k)
        rows.append(("", ""))
        rows.append(("--- Acceptance Distribution ---", ""))
        for bucket, cnt in sorted(dist.items()):
            label = f"  {bucket} accepted" if bucket < k else f"  {bucket}+ accepted"
            rows.append((label, f"{cnt}"))

    max_label = max(len(r[0]) for r in rows)
    for label, value in rows:
        if not label and not value:
            print()
        elif not value:
            print(f"  {label}")
        else:
            print(f"  {label:<{max_label}}  {value}")

    print(sep)
    print()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(metrics: list[RequestMetrics], path: str, mode: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "request_id",
        "mode",
        "prompt_tokens",
        "completion_tokens",
        "ttft_ms",
        "total_decode_ms",
        "tpt_ms",
    ]
    if mode == "eagle-3":
        fieldnames += [
            "num_spec_steps",
            "total_drafted",
            "total_accepted",
            "draft_head_time_ms",
            "verify_time_ms",
        ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics:
            row = {
                "request_id": m.request_id,
                "mode": mode,
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "ttft_ms": f"{m.ttft_ms:.3f}",
                "total_decode_ms": f"{m.total_decode_ms:.3f}",
                "tpt_ms": f"{m.tpt_ms:.3f}",
            }
            if mode == "eagle-3":
                row.update({
                    "num_spec_steps": m.num_spec_steps,
                    "total_drafted": m.total_drafted,
                    "total_accepted": m.total_accepted,
                    "draft_head_time_ms": f"{m.draft_head_time_ms:.3f}",
                    "verify_time_ms": f"{m.verify_time_ms:.3f}",
                })
            writer.writerow(row)

    print(f"Per-request metrics saved to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="nano-vllm benchmark: AR vs EAGLE-3 speculative decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["ar", "eagle-3"],
        required=True,
        help="Decoding mode: 'ar' for autoregressive, 'eagle-3' for speculative",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/Qwen3-0.6B/",
        help="Path to the base model",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="./models/Qwen3-0.6B-draft/",
        help="Path to the EAGLE-3 draft model (only used when --mode eagle-3)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
        help="Path to ShareGPT dataset JSON",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of prompts to evaluate",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--draft-tokens",
        type=int,
        default=5,
        help="Number of draft tokens per speculative step (EAGLE-3 only)",
    )
    parser.add_argument(
        "--tree-size",
        type=int,
        default=None,
        help="Alias for --draft-tokens (EAGLE-3 only)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum context length",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (use eager execution)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="logs/benchmark_results.csv",
        help="Path for per-request CSV output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt sampling",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup requests before benchmark",
    )

    args = parser.parse_args()

    # --tree-size is an alias for --draft-tokens
    if args.tree_size is not None:
        args.draft_tokens = args.tree_size

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print(f"[benchmark] mode={args.mode}, samples={args.num_samples}, "
          f"max_new_tokens={args.max_new_tokens}, temperature={args.temperature}")

    # Build engine kwargs
    engine_kwargs = {
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
    }

    if args.mode == "eagle-3":
        engine_kwargs["draft_model"] = args.draft_model
        engine_kwargs["num_speculative_tokens"] = args.draft_tokens
        print(f"[benchmark] draft_model={args.draft_model}, "
              f"draft_tokens={args.draft_tokens}")

    print(f"[benchmark] Loading model: {args.model}")
    engine = LLMEngine(args.model, **engine_kwargs)
    tokenizer = engine.tokenizer

    # Print environment info
    print(f"[benchmark] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[benchmark] CUDA: {torch.version.cuda}")
    print(f"[benchmark] PyTorch: {torch.__version__}")

    # Load dataset
    print(f"[benchmark] Loading prompts from: {args.data}")
    prompts = load_sharegpt_prompts(
        data_path=args.data,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    print(f"[benchmark] Loaded {len(prompts)} prompts")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        ignore_eos=False,
    )

    # Warmup
    if args.warmup > 0:
        print(f"[benchmark] Warmup: {args.warmup} request(s)...")
        warmup_prompts = prompts[:args.warmup]
        warmup_sp = SamplingParams(
            temperature=args.temperature,
            max_tokens=16,
            ignore_eos=True,
        )
        engine.generate(warmup_prompts, warmup_sp, use_tqdm=False)
        torch.cuda.synchronize()

    # Run benchmark
    print(f"[benchmark] Running benchmark...")
    bench = BenchmarkEngine(engine, args.mode, args.draft_tokens)
    per_request = bench.run(prompts, sampling_params)
    summary = bench.summary

    # Print results
    print_summary(summary, args.draft_tokens)

    # Save CSV
    save_csv(per_request, args.output_csv, args.mode)


if __name__ == "__main__":
    main()
