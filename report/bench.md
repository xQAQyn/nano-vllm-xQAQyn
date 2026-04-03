# nano-vllm Benchmark Report: AR vs EAGLE-3 Speculative Decoding

## Overview

This benchmark suite evaluates the performance of the nano-vllm inference engine under two decoding strategies:

- **Autoregressive (AR):** The standard token-by-token generation approach where each token requires one full base model forward pass.
- **EAGLE-3 Speculative Decoding:** A draft-then-verify strategy that uses a lightweight draft model to propose multiple candidate tokens, which the base model verifies in a single forward pass. This amortizes the cost of memory-bound decoding steps and mitigates the **memory wall** -- the bottleneck where inference speed is limited by memory bandwidth rather than compute.

The benchmark collects throughput, latency, VRAM usage, and EAGLE-3-specific acceptance metrics from real conversational prompts drawn from the ShareGPT dataset.

## Environment Setup

| Component         | Value                |
|-------------------|----------------------|
| GPU               | *(fill in)*          |
| CUDA Version      | *(fill in)*          |
| PyTorch Version   | *(fill in)*          |
| Base Model        | Qwen3-0.6B           |
| Draft Model       | Qwen3-0.6B-draft     |
| Max Context Length | 4096                 |
| Dataset           | ShareGPT V3 (filtered) |

## Usage Instructions

### Prerequisites

```bash
# Install dependencies
pip install -e .
# Ensure models are available under ./models/
# Ensure dataset is at data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json
```

### Running the AR Baseline

```bash
python benchmark.py \
    --mode ar \
    --model ./models/Qwen3-0.6B/ \
    --num-samples 50 \
    --max-new-tokens 256 \
    --temperature 0.6
```

### Running the EAGLE-3 Benchmark

```bash
python benchmark.py \
    --mode eagle-3 \
    --model ./models/Qwen3-0.6B/ \
    --draft-model ./models/Qwen3-0.6B-draft/ \
    --num-samples 50 \
    --max-new-tokens 256 \
    --temperature 0.6 \
    --draft-tokens 5
```

### Full CLI Reference

| Argument            | Default                     | Description |
|---------------------|-----------------------------|-------------|
| `--mode`            | *(required)*                | `ar` or `eagle-3` |
| `--model`           | `./models/Qwen3-0.6B/`     | Path to the base model |
| `--draft-model`     | `./models/Qwen3-0.6B-draft/` | Path to the EAGLE-3 draft model |
| `--data`            | `data/ShareGPT_V3_...json` | Path to ShareGPT dataset |
| `--num-samples`     | `50`                        | Number of prompts to evaluate |
| `--max-new-tokens`  | `256`                       | Max generation length per request |
| `--temperature`     | `0.6`                       | Sampling temperature |
| `--draft-tokens`    | `5`                         | Draft tokens per speculative step |
| `--tree-size`       | *(alias for --draft-tokens)* | Alternative name for draft-tokens |
| `--max-model-len`   | `4096`                      | Maximum sequence length |
| `--enforce-eager`   | `false`                     | Disable CUDA graphs |
| `--output-csv`      | `logs/benchmark_results.csv`| CSV output path |
| `--seed`            | `42`                        | Random seed for prompt sampling |
| `--warmup`          | `2`                         | Warmup requests before benchmark |

## Metrics Definition

### Universal Metrics (AR and EAGLE-3)

| Metric | Definition |
|--------|------------|
| **Throughput (TPS)** | Total completion tokens divided by total wall-clock time. Measures end-to-end generation speed in tokens per second. |
| **TTFT (Time to First Token)** | GPU time for the prefill (prompt encoding) phase. This is the latency a user experiences before seeing the first output token. Measured with `torch.cuda.Event` for accurate GPU timing. |
| **TPT (Time Per Output Token)** | Average GPU time per generated output token during the decode phase. Lower TPT indicates more efficient decoding. |
| **VRAM Peak** | Peak GPU memory allocated during the benchmark run, measured via `torch.cuda.max_memory_allocated()`. Includes model weights, KV cache, activations, and any speculative buffers. |

### EAGLE-3 Exclusive Metrics

| Metric | Definition |
|--------|------------|
| **Mean Acceptance Length (tau)** | The average number of drafted tokens accepted per base model verification step. If the draft model proposes *k* tokens and on average *tau* are accepted, each base model forward pass produces *tau + 1* tokens (including the bonus token). Higher tau means better draft quality and greater speedup. |
| **Acceptance Rate Distribution** | A histogram showing how often exactly 0, 1, 2, ..., *k* draft tokens were accepted in a single speculative step. Reveals the reliability of the draft model -- a distribution skewed right indicates a well-matched draft model. |
| **Draft Head Overhead** | Average GPU time spent executing the EAGLE-3 draft head (feature fusion + autoregressive draft generation) per speculative step. This is the additional latency introduced by speculation. For EAGLE-3 to be beneficial, this overhead must be significantly less than the time saved by batched verification. |
| **Base Model Verification Time** | Average GPU time for the base model to verify all *k* draft candidates in a single forward pass. The key insight: verifying *k+1* positions in one pass costs roughly the same as generating 1 token autoregressively (both are memory-bandwidth bound), so accepted drafts are essentially "free". |
| **Tree Efficiency** | The ratio `Accepted_Tokens / Total_Drafted_Tokens`. Measures what fraction of the draft model's compute was useful. A score of 1.0 means every drafted token was accepted; typical values range from 0.4 to 0.8 depending on task difficulty and draft model quality. |

### How EAGLE-3 Mitigates the Memory Wall

LLM inference is fundamentally **memory-bandwidth bound** during the decode phase: generating each token requires reading the entire model's weights from GPU memory, but performs very little arithmetic per byte transferred. This means a single token decode step leaves most of the GPU's compute units idle.

EAGLE-3 addresses this by:

1. **Draft phase:** A small, cheap model generates *k* candidate tokens with minimal memory overhead (single transformer layer).
2. **Verify phase:** The base model processes all *k+1* positions in a single forward pass. Since the bottleneck is reading model weights (which happens once regardless of batch size), verifying *k+1* tokens costs nearly the same as generating 1.
3. **Net effect:** Each base model forward pass now yields *tau + 1* tokens instead of 1, achieving a speedup proportional to the mean acceptance length.

The **Mean Acceptance Length (tau)** directly quantifies this speedup factor, while **Tree Efficiency** measures how much draft compute is wasted on rejected tokens.

## Results

### Summary Comparison

| Metric                    | AR Baseline | EAGLE-3  | Speedup |
|---------------------------|-------------|----------|---------|
| Throughput (TPS)          | *(fill in)* | *(fill in)* | *(fill in)* |
| TTFT (ms)                 | *(fill in)* | *(fill in)* | --      |
| TPT (ms)                  | *(fill in)* | *(fill in)* | *(fill in)* |
| VRAM Peak (MB)            | *(fill in)* | *(fill in)* | delta: *(fill in)* |
| Mean Acceptance Length     | --          | *(fill in)* | --      |
| Tree Efficiency           | --          | *(fill in)* | --      |
| Avg Draft Head Time (ms)  | --          | *(fill in)* | --      |
| Avg Verify Time (ms)      | --          | *(fill in)* | --      |

### Acceptance Distribution (EAGLE-3)

| Tokens Accepted | Count |
|-----------------|-------|
| 0               | *(fill in)* |
| 1               | *(fill in)* |
| 2               | *(fill in)* |
| 3               | *(fill in)* |
| 4               | *(fill in)* |
| 5               | *(fill in)* |

### Per-Request Details

Detailed per-request metrics are saved to `logs/benchmark_results.csv` after each run. The CSV includes prompt/completion token counts, TTFT, TPT, and (for EAGLE-3 mode) per-request acceptance statistics.

## Reproducing Results

```bash
# 1. Run AR baseline
python benchmark.py --mode ar --num-samples 50 --max-new-tokens 256 \
    --output-csv logs/ar_results.csv

# 2. Run EAGLE-3
python benchmark.py --mode eagle-3 --num-samples 50 --max-new-tokens 256 \
    --draft-tokens 5 --output-csv logs/eagle3_results.csv

# 3. Compare outputs
# Both runs use the same seed (42) so the same prompts are selected.
```
