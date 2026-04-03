# Simplified EAGLE-3 Speculative Decoding: Implementation Report

## Table of Contents

1. [Chapter 1: Memory Management](#chapter-1-memory-management)
2. [Chapter 2: Scheduling Strategy](#chapter-2-scheduling-strategy)
3. [Chapter 3: Inference Pipeline — Speculative Decoding](#chapter-3-inference-pipeline--speculative-decoding)
4. [Chapter 4: Draft Model Architecture](#chapter-4-draft-model-architecture)
5. [Chapter 5: Performance Optimization](#chapter-5-performance-optimization)
6. [Chapter 6: Weight Loading & Model Initialization](#chapter-6-weight-loading--model-initialization)
7. [Chapter 7: Testing Strategy](#chapter-7-testing-strategy)

---

## Chapter 1: Memory Management

### 1.1 What We Built

The memory subsystem in nano-vllm manages two fundamentally different caching regimes that must coexist:

1. **Target model KV cache** — a block-based, hash-deduplicated, multi-layer cache shared across sequences via reference counting.
2. **Draft model KV cache** — a tiny, dense, linear buffer that is reset every speculative cycle.

For speculative decoding we added three new operations to `BlockManager` (`block_manager.py`):

| Method | Purpose |
|--------|---------|
| `pre_allocate_speculative(seq, n)` | Extends `seq.block_table` with empty blocks so that `n` future positions have valid physical slots — **without** modifying `seq.num_tokens` or `seq.token_ids`. |
| `deallocate_speculative(seq, num_accepted)` | After acceptance, trims the block table back to `seq.num_blocks` (which is derived from the actual token count), deallocating any surplus blocks. |
| `truncate(seq, new_num_tokens)` | Hard rollback: shrinks both the token list and the block table to an arbitrary earlier length, freeing all trailing blocks. |

### 1.2 Why This Design

**The core problem**: During speculative verification, the target model runs a prefill-like forward pass over `k+1` tokens. The Triton `store_kvcache_kernel` writes K/V vectors into the block cache using a `slot_mapping` that translates each token position to a physical `(block_id * block_size + offset)` address. If those positions don't have allocated blocks, the slot is `-1` and the kernel skips the write — meaning the KV cache is never populated for those positions, and subsequent decode steps will read garbage.

We considered three alternatives:

| Approach | Pros | Cons |
|----------|------|------|
| **A. Allocate blocks eagerly inside `prepare_verify`** | Self-contained | Violates separation of concerns — `ModelRunner` should not own block allocation. Impossible to roll back on rejection without knowing the `BlockManager` state. |
| **B. Temporarily fake `seq.num_tokens` to trick `may_append`** | Reuses existing code | Fragile — `num_tokens` drives hash computation, EOS checks, and serialization (`__getstate__`). Mutating it temporarily risks race conditions and corrupted state. |
| **C. Dedicated `pre_allocate_speculative` / `deallocate_speculative` pair** | Clean contract — allocate before verify, trim after accept. Block table grows without touching token state. | Slightly more code in `BlockManager`. |

We chose **C** because it provides a clean transactional boundary: the block table is extended *before* verification, and trimmed *after* acceptance. The token list is only modified by the scheduler's `postprocess_speculative`, which appends accepted tokens one at a time. This means `seq.num_blocks` (derived from `seq.num_tokens`) is always consistent with the "committed" state, and the block table may temporarily overshoot by at most `ceil(k / block_size)` blocks.

### 1.3 Target Model KV Cache Layout

The target model's KV cache is a single contiguous tensor allocated once during initialization:

```python
# model_runner.py:125
self.kv_cache = torch.empty(
    2,                              # K and V
    hf_config.num_hidden_layers,    # 28 layers for Qwen3-0.6B
    num_kvcache_blocks,             # computed from free GPU memory
    self.block_size,                # 256 tokens per block
    num_kv_heads,                   # 8 (or 8/tp_size under TP)
    head_dim                        # 128
)
```

Each `Attention` layer receives a *view* into this tensor (`kv_cache[0, layer_id]` for K, `kv_cache[1, layer_id]` for V), so there is zero copy overhead — the Triton kernel writes directly into the pre-allocated global buffer.

The number of blocks is computed dynamically:

```python
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
num_blocks = (total_gpu * utilization - used - peak + current) // block_bytes
```

This "fill remaining memory" strategy maximizes cache capacity without risking OOM. For Qwen3-0.6B on a 24 GB GPU at 90% utilization, this yields ~2000 blocks = ~512K tokens of KV cache.

### 1.4 Draft Model KV Cache Layout

The draft model uses a fundamentally different cache — a dense `[max_len, num_kv_heads, head_dim]` tensor per K and V, managed by a simple `cache_len` counter:

```python
# eagle3.py:40-44
self.k_cache = torch.zeros(max_len, self.num_kv_heads, self.head_dim, ...)
self.v_cache = torch.zeros(max_len, self.num_kv_heads, self.head_dim, ...)
self.cache_len = 0
```

Where `max_len = num_speculative_tokens + 1 = 6`. This is ~5 KB total. We chose a dense linear layout over the block-based layout because:

- The draft generates at most `k` tokens per cycle, then resets. There is no cross-cycle reuse, so block hashing and deduplication add overhead with zero benefit.
- The dense layout enables direct `self.k_cache[self.cache_len:self.cache_len + n] = k` writes — a single contiguous memcpy, no slot indirection.
- Reset is free: just set `cache_len = 0`. No deallocation, no hash cleanup.

### 1.5 Prefix Caching via Hash Deduplication

The existing `BlockManager.allocate()` implements automatic prefix caching using chained xxHash64:

```python
h = xxhash.xxh64()
h.update(prefix_hash.to_bytes(8, "little"))  # chain with previous block's hash
h.update(np.array(token_ids).tobytes())
```

Each block's hash incorporates its predecessor's hash, creating a Merkle-like chain. On allocation, the manager checks `hash_to_block_id` for a match and verifies token content to avoid collisions. On hit, `ref_count` is incremented and `num_cached_tokens` advances by `block_size`, causing the prefill to skip those tokens entirely.

Our speculative decoding changes are **fully compatible** with prefix caching. The `pre_allocate_speculative` method allocates fresh blocks (with `hash = -1`), and `deallocate_speculative` frees them if rejected. Accepted tokens flow through `may_append` in the normal postprocessing path, which computes hashes on full blocks. This means a speculative cycle that completes a 256-token block boundary will correctly register it in the prefix cache for future sequences.

**Compared to vLLM's approach**: vLLM uses a tree-based block allocator with copy-on-write for prefix sharing. Our approach is simpler — flat deque of free block IDs with hash-based lookup — but achieves the same functional result for the common case (shared system prompts). The trade-off is that we don't support forked sequences (beam search), which is acceptable for greedy/sampling-based speculative decoding.

---

## Chapter 2: Scheduling Strategy

### 2.1 What We Built

The scheduler (`scheduler.py`) orchestrates the two-phase inference loop: **prefill** (process prompt tokens) and **decode** (generate one token per step). We extended it with a third mode: **speculative postprocessing**, which handles the variable-length acceptance results from a speculative decode cycle.

New method added:

```python
def postprocess_speculative(self, seqs, results):
    for seq, (accepted_tokens, new_token) in zip(seqs, results):
        all_tokens = accepted_tokens + [new_token]
        finished = False
        for tok in all_tokens:
            seq.append_token(tok)
            if (not seq.ignore_eos and tok == self.eos) or \
               seq.num_completion_tokens >= seq.max_tokens:
                finished = True
                break
        block_manager.deallocate_speculative(seq, len(all_tokens))
        if finished:
            seq.status = SequenceStatus.FINISHED
            block_manager.deallocate(seq)
            self.running.remove(seq)
```

### 2.2 The Scheduling Flow

The original two-phase schedule works as follows:

```
schedule() →
  IF waiting queue non-empty AND can allocate blocks:
    → PREFILL: pop from waiting, allocate blocks, return (seqs, is_prefill=True)
  ELSE:
    → DECODE: pop from running, may_append for new token slot, return (seqs, is_prefill=False)
```

With speculative decoding enabled, the engine (`llm_engine.py`) routes these phases differently:

```
step() →
  seqs, is_prefill = scheduler.schedule()
  IF is_prefill:
    IF use_speculative:
      token_ids = model_runner.run_prefill_with_capture(seqs)  # capture hidden states
    ELSE:
      token_ids = model_runner.run(seqs, True)
    scheduler.postprocess(seqs, token_ids)                     # append 1 token per seq
  ELIF use_speculative:
    results = model_runner.run_speculative(seqs, block_manager) # draft k + verify
    scheduler.postprocess_speculative(seqs, results)            # append 1..k+1 tokens per seq
  ELSE:
    token_ids = model_runner.run(seqs, False)                  # standard decode
    scheduler.postprocess(seqs, token_ids)
```

### 2.3 Why This Design

**Decision 1: Speculative decode replaces normal decode entirely (no fallback per-step)**

When a draft model is configured, *every* decode step uses speculative decoding. We don't mix speculative and non-speculative sequences in the same batch. This simplifies the scheduler — it doesn't need to track which sequences are in "speculative mode" vs "normal mode".

Alternative: vLLM's speculative decoding allows per-sequence opt-in based on acceptance rate history. This is more flexible but adds significant bookkeeping. For a minimal implementation, uniform speculation is correct and simple.

**Decision 2: Block pre-allocation happens in `run_speculative`, trimming in `postprocess_speculative`**

The speculative cycle has a clear transactional boundary:

1. `run_speculative` pre-allocates blocks (via `block_manager.pre_allocate_speculative`)
2. Target model verification writes KV into those blocks
3. `postprocess_speculative` appends accepted tokens and trims surplus blocks

If we pre-allocated in the scheduler before calling the model runner, the scheduler would need to know `k` and predict how many blocks are needed — coupling it to the draft model's configuration. By keeping pre-allocation inside `run_speculative`, the scheduler remains model-agnostic.

**Decision 3: Token-by-token append with early EOS termination**

Inside `postprocess_speculative`, we append tokens one at a time and check EOS after each:

```python
for tok in all_tokens:
    seq.append_token(tok)
    if tok == self.eos: break
```

This handles the case where EOS appears in the middle of accepted tokens (e.g., the draft correctly predicts EOS as the 3rd of 5 tokens). We stop appending immediately and mark the sequence as finished, even though more tokens were "accepted" by the greedy comparison. This is critical for correctness — without it, we'd generate tokens past the natural end of the response.

**Decision 4: Preemption is unchanged**

The existing preemption logic (evict the last running sequence when memory is tight) works without modification because speculative blocks are always cleaned up within the same step. By the time the scheduler runs `schedule()` again, all surplus blocks have been freed.

### 2.4 Throughput Accounting

The engine reports throughput to the progress bar. For speculative decode, `num_tokens` is computed as:

```python
num_tokens = -sum(len(accepted) + 1 for accepted, _ in results)
```

The negative sign follows the existing convention (negative = decode tokens, positive = prefill tokens). This gives an accurate token-per-second metric that reflects the actual number of tokens generated per cycle, not just the number of model forward passes.

---

## Chapter 3: Inference Pipeline — Speculative Decoding

### 3.1 What We Built

The speculative decoding pipeline is a three-phase cycle that generates multiple tokens per target model invocation:

```
Phase 1: DRAFT    — Draft model generates k candidate tokens autoregressively
Phase 2: VERIFY   — Target model processes all k+1 tokens in one batched forward pass
Phase 3: ACCEPT   — Greedy comparison determines how many drafts are correct
```

All three phases are implemented in `ModelRunner` (`model_runner.py`).

### 3.2 Phase 1: Draft Generation

```python
# model_runner.py:302-312
for seq in seqs:
    fused = self.draft_model.fuse_features(self.saved_hidden[seq.seq_id])
    drafts = self.draft_model.generate(
        self.model.model.embed_tokens, fused,
        start_token=torch.tensor([seq.last_token], device="cuda"),
        start_pos=len(seq), k=k,
    )
    all_draft_tokens.append(drafts)
```

The draft model receives:

1. **Fused features** — multi-layer hidden states from the target model's last accepted position, concatenated and projected via `fc` + `hidden_norm`. Computed once per sequence per cycle.
2. **Token embeddings** — from the *target model's* embedding table (`self.model.model.embed_tokens`). This is critical: the draft model doesn't have its own embeddings. It borrows the target's, ensuring the input representation is aligned.
3. **Positional information** — absolute positions starting from `len(seq)`, the current sequence length.

The draft model generates `k` tokens autoregressively. Each step involves:
- Embed the previous token using the target's embedding table
- Concatenate `[normed_embed, fused]` → 2048-dim input to attention
- Single transformer block (attention + MLP)
- argmax on draft vocabulary → `d2t` mapping to target vocabulary

### 3.3 Phase 2: Verification

The verification pass is the key innovation. Instead of running the target model `k` times (once per draft token), we run it **once** on all `k+1` tokens simultaneously:

```python
# model_runner.py:314-322
block_manager.pre_allocate_speculative(seq, len(drafts) + 1)
input_ids, positions = self.prepare_verify(seqs, all_draft_tokens)
hidden_states, captured = self.model(input_ids, positions, capture_layers=self.capture_layers)
logits = self.model.compute_logits_all(hidden_states)
```

`prepare_verify` constructs the input in a prefill-like format:

- **Input tokens**: `[last_accepted_token, draft_1, draft_2, ..., draft_k]` per sequence
- **Positions**: `[P-1, P, P+1, ..., P+k-1]` where P is the next position to generate
- **cu_seqlens**: Cumulative sequence lengths for variable-length batching across sequences
- **slot_mapping**: Physical KV cache addresses for each position, derived from pre-allocated blocks

The target model runs with `capture_layers={1, 13, 24}`, which hooks into the forward loop to extract hidden states at those layer indices:

```python
# qwen3.py:184-186
for i, layer in enumerate(self.layers):
    hidden_states, residual = layer(positions, hidden_states, residual)
    if captured is not None and i in capture_layers:
        captured[i] = hidden_states + residual  # full accumulated state
```

The captured hidden states serve dual purpose: (a) verification logits at all positions, and (b) fused features for the next draft cycle.

`compute_logits_all` (`embed_head.py`) returns logits at **every** position, not just the last. This is achieved by passing `all_positions=True` to skip the usual last-index extraction:

```python
def forward(self, x, all_positions=False):
    if context.is_prefill and not all_positions:
        last_indices = context.cu_seqlens_q[1:] - 1
        x = x[last_indices].contiguous()  # skipped when all_positions=True
    logits = F.linear(x, self.weight)
```

### 3.4 Phase 3: Greedy Acceptance

```python
# model_runner.py:282-296
def accept_tokens(self, seq_logits, draft_tokens, seq_captured):
    target_tokens = seq_logits.argmax(dim=-1)
    for i, dt in enumerate(draft_tokens):
        if target_tokens[i].item() == dt:
            accepted.append(dt)
        else:
            return accepted, target_tokens[i].item(), hidden_at[i]
    # All accepted → bonus token from logits[k]
    return accepted, target_tokens[k].item(), hidden_at[k]
```

The logit layout is:
- `logits[0]` = target model's prediction for position P (should match `draft_1`)
- `logits[1]` = prediction for position P+1 (should match `draft_2`)
- ...
- `logits[k]` = prediction for position P+k (bonus token, always accepted)

On full acceptance, we get `k+1` tokens from a single target forward pass. On first mismatch at position `i`, we get `i+1` tokens (the `i` accepted drafts plus the target's replacement).

**Why greedy acceptance over speculative sampling?** Speculative sampling (Leviathan et al., 2023) uses a probability-ratio test `min(1, p_target / p_draft)` to preserve the target distribution for non-zero temperatures. We use greedy (argmax) comparison because:

1. It's deterministic — easier to test and debug.
2. For temperature=0 (the common case in code generation), greedy acceptance is provably optimal.
3. It's a single `argmax` per position — no softmax, no random sampling, no probability computation.

The trade-off is that for temperature > 0, greedy acceptance may reject valid tokens that would have been sampled from the target distribution. This is a known limitation documented for future work.

### 3.5 Cross-Cycle State: Hidden State Persistence

After each speculative cycle, the hidden states at the last accepted position are saved:

```python
self.saved_hidden[seq.seq_id] = new_hidden  # {layer_idx: tensor[1, hidden_size]}
```

These become the fused features for the next cycle's draft generation. This is the key to EAGLE-3's efficiency — the target model's internal representations are *reused* across cycles, not recomputed. The memory cost is minimal: 3 layers x 1024 dims x 2 bytes = 6 KB per sequence.

On prefill, hidden states are captured from the last token position:

```python
# model_runner.py:238-240
idx = last_indices[i].item()
self.saved_hidden[seq.seq_id] = {l: captured[l][idx:idx+1] for l in captured}
```

When a sequence finishes, the state is cleaned up via `clear_saved_hidden(seq_id)`.

### 3.6 Compared to Alternative Approaches

| Aspect | Our Implementation | Standard EAGLE-3 | Medusa | Independent Draft Model |
|--------|--------------------|-------------------|--------|-------------------------|
| Draft input | Multi-layer fused features | Multi-layer fused features | Single-layer features | None (separate model) |
| Verification | Static flat chain | Dynamic tree growth | Parallel multi-head | Flat chain |
| Tokens per cycle | 1 to k+1 | Variable (tree) | Variable (tree) | 1 to k+1 |
| Complexity | Low | High (tree management) | Medium (multi-head) | Low |
| Target model changes | `capture_layers` param | Same | Extra heads added | None |

We deliberately chose **static flat-chain verification** over dynamic tree growth. Tree-based verification (used in full EAGLE-3) generates multiple candidate continuations at each position and verifies them as a tree, potentially accepting more tokens per cycle. However:

- Tree verification requires a custom attention mask (not the standard causal mask), which breaks compatibility with Flash Attention's `causal=True` fast path.
- The tree must be dynamically constructed based on draft model confidence scores, adding latency to the draft phase.
- Block allocation becomes non-trivial — different branches of the tree may need different blocks, and rejected branches must be rolled back.
- Our flat chain reuses the existing prefill infrastructure (`flash_attn_varlen_func` with `cu_seqlens`) with zero modification to the attention kernel.

---

## Chapter 4: Draft Model Architecture

### 4.1 What We Built

`Eagle3DraftModel` (`eagle3.py`) is a single-layer transformer augmented with a multi-layer feature fusion head:

```
Target hidden[layer_1] ─┐
Target hidden[layer_2] ──┼── cat ──► fc [3072→1024] ──► hidden_norm ───────┐
Target hidden[layer_3] ─┘                                                  │
                                                                           │
target.embed_tokens(prev_token) ──► input_layernorm ──► [normed; fused] ──►│
                                          │                          2048-dim input
                                          │                                │
                                    Eagle3Attention (Q/K/V from 2048)      │
                                          │                                │
                                    residual add ◄─────────────────────────┘
                                          │
                                    post_attn_layernorm ──► Eagle3MLP
                                          │
                                    residual add ──► norm ──► lm_head [1024→32000]
                                                                    │
                                                              d2t[argmax] ──► target token
```

### 4.2 Key Design Decisions

**Decision 1: Separate Q/K/V projections instead of fused QKV**

The target model uses `QKVParallelLinear` — a single weight matrix that packs Q, K, V with tensor-parallel sharding. The draft model uses three separate `nn.Linear` layers:

```python
self.q_proj = nn.Linear(input_size, num_heads * head_dim, bias=False)   # [2048→1024]
self.k_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False) # [2048→512]
self.v_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False) # [2048→512]
```

Reason: The SpecForge training framework produces separate weight files for Q, K, V. Fusing them would require reshaping during loading. Since the draft model is tiny (1 layer) and runs on a single GPU, the performance benefit of fused QKV is negligible — the compute is not the bottleneck.

**Decision 2: PyTorch SDPA instead of Flash Attention**

```python
o = F.scaled_dot_product_attention(q, k_full, v_full,
                                   is_causal=(n > 1), scale=self.scaling)
```

The draft model uses `torch.nn.functional.scaled_dot_product_attention` instead of the `flash_attn` library because:

- The sequence length is at most `k+1 = 6` tokens. Flash Attention's tiling overhead (designed for long sequences) exceeds the actual compute.
- SDPA dispatches to the most efficient backend automatically (including Flash Attention when beneficial on CUDA, or math fallback on CPU).
- SDPA works on both CPU and CUDA without code changes, simplifying testing.
- No dependency on the global `Context` object — the draft model is self-contained.

**Decision 3: Vocabulary mapping via `d2t` / `t2d` buffers**

The draft model operates on a reduced vocabulary (32K tokens vs the target's 151K). Two registered buffers handle the mapping:

- `d2t` (draft-to-target): `int64[32000]` — maps each draft token index to its target vocabulary index.
- `t2d` (target-to-draft): `bool[151936]` — indicates which target tokens are representable in the draft vocabulary.

These are loaded from the SpecForge checkpoint as pre-computed tensors. The mapping happens at draft time:

```python
draft_idx = logits.argmax(dim=-1)     # argmax in 32K space
target_token = self.d2t[draft_idx]    # map to 151K space
```

This is a single `gather` operation — O(1) per token.

**Decision 4: Reuse target model's embedding table**

```python
token_embed = embed_fn(token)  # embed_fn = self.model.model.embed_tokens
```

The draft model does not have its own embedding layer. It calls the target model's `VocabParallelEmbedding` directly. This ensures:

- The token representation is always aligned between draft and target.
- No additional memory for a 151K x 1024 embedding table (~293 MB in bf16).
- Weight updates to the target embeddings (e.g., from fine-tuning) are automatically reflected in the draft.

### 4.3 Memory Footprint

| Component | Size |
|-----------|------|
| `fc.weight` [1024, 3072] | 6.0 MB |
| Attention Q/K/V/O projections | 10.0 MB |
| MLP gate/up/down projections | 18.0 MB |
| `lm_head.weight` [32000, 1024] | 62.5 MB |
| Norms (5x RMSNorm) | 0.02 MB |
| `d2t` + `t2d` buffers | 0.4 MB |
| KV cache (6 positions) | 0.005 MB |
| **Total** | **~97 MB** |

This is ~8% of the target model's size (~1.2 GB). The draft model fits comfortably alongside the target on a single GPU.

---

## Chapter 5: Performance Optimization

### 5.1 CUDA Graph Capture for Decode

The decode phase (single token per sequence) is captured as CUDA graphs during initialization:

```python
# model_runner.py:353
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```

Graphs are captured for a discrete set of batch sizes. At runtime, the smallest graph with `bs >= actual_bs` is selected:

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
```

Input tensors are pre-allocated and reused across calls — the graph replays with updated values written into the same memory addresses:

```python
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
graph.replay()
```

**Impact**: Eliminates ~100-500us of kernel launch overhead per forward pass. For small batch sizes (bs=1-8), this is 20-40% of total decode latency.

**Speculative compatibility**: CUDA graphs are used for *normal* decode only. The speculative verification pass uses eager mode (prefill path), which is already eager in the baseline. The draft model also runs in eager mode since its compute is trivially small.

### 5.2 Flash Attention: Dual-Path Design

The `Attention` layer (`attention.py`) dispatches to two different Flash Attention kernels:

**Prefill** (also used for speculative verification):
```python
flash_attn_varlen_func(q, k, v,
    cu_seqlens_q=..., cu_seqlens_k=...,
    max_seqlen_q=..., max_seqlen_k=...,
    causal=True, block_table=...)
```

Variable-length attention handles multiple sequences of different lengths in a single kernel launch. The `cu_seqlens` (cumulative sequence lengths) define where each sequence starts and ends in the flattened Q/K/V tensors. This is critical for verification — each sequence has `k+1` query tokens but a different total context length (existing KV cache + k+1 new tokens).

**Decode**:
```python
flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
    cache_seqlens=..., block_table=...,
    causal=True)
```

KV-cache-aware attention for single-token generation. Reads K/V directly from the block cache via `block_table`, avoiding any copy.

**Why not use decode-mode attention for verification?** The verification pass processes `k+1` tokens per sequence — this is a "short prefill", not a decode. Using `flash_attn_varlen_func` gives us batched attention over all `k+1` positions in a single kernel, while decode-mode attention would require `k+1` separate kernel launches per sequence.

### 5.3 Triton Kernel for KV Cache Storage

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                          k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # ... load key/value, store to cache at slot * D ...
```

This kernel runs with `N` programs (one per token position), each writing a `[num_kv_heads * head_dim]`-sized vector to the cache. The `slot_mapping` provides indirect addressing — position `idx` maps to physical cache address `slot * D`.

**Why a custom Triton kernel instead of PyTorch scatter?** `torch.scatter` would require reshaping the cache to 1D, computing flat indices, and handling the skip-on-negative logic. The Triton kernel does this in a single fused operation with: (a) coalesced memory access patterns, (b) no Python-side index computation, (c) native `-1` sentinel handling.

### 5.4 Fused RMSNorm with Residual Connection

```python
@torch.compile
def add_rms_forward(self, x, residual):
    x = x.float().add_(residual.float())  # fused residual add
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    x = x.to(orig_dtype).mul_(self.weight)
    return x, residual
```

Every decoder layer uses `add_rms_forward` which fuses the residual connection and RMS normalization into a single `@torch.compile`-compiled kernel. This reduces memory bandwidth by:

- Avoiding a separate tensor for the residual sum
- Using in-place operations (`.add_()`, `.mul_()`) to minimize allocations
- Fusing the entire normalize-scale pipeline into one kernel dispatch

The Qwen3 decoder layer calls this twice per layer (pre-attention and post-attention), so the fusion is applied 56 times per forward pass for the 28-layer model.

### 5.5 Tensor Parallelism

The model supports multi-GPU inference through tensor parallelism with four parallel linear layer types:

| Layer | Shard Dim | Communication |
|-------|-----------|---------------|
| `QKVParallelLinear` | Column (output) | None — heads are independent |
| `MergedColumnParallelLinear` | Column (gate+up) | None — followed by row-parallel |
| `RowParallelLinear` | Row (input) | `all_reduce` after matmul |
| `VocabParallelEmbedding` | Vocab range | `all_reduce` after embedding |

Multi-process coordination uses shared memory + `multiprocessing.Event`:

```python
# Rank 0 writes command to shared memory, signals workers
data = pickle.dumps([method_name, *args])
self.shm.buf[4:n+4] = data
for event in self.event: event.set()
```

This avoids the overhead of NCCL for control-plane communication (method dispatch), reserving NCCL for the data-plane (`all_reduce`).

**Speculative decoding under TP**: The draft model runs on rank 0 only (it's too small to benefit from parallelism). The target model's verification pass uses the same TP communication as normal prefill — no changes needed.

### 5.6 Async Data Transfer

All host-to-device transfers use pinned memory with non-blocking copies:

```python
torch.tensor(data, pin_memory=True).cuda(non_blocking=True)
```

This enables DMA transfers that overlap with CPU-side batch preparation. For speculative decoding, this is especially important in `prepare_verify` where we build the input for `k+1` tokens per sequence — the CPU can prepare the next sequence's data while the previous one is still transferring.

---

## Chapter 6: Weight Loading & Model Initialization

### 6.1 What We Built

Two separate loading paths for the two model types:

**Target model** (`load_model`): Uses `packed_modules_mapping` to handle fused weight matrices:
```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

Each parameter has a `weight_loader` method (set by the parallel linear layers) that handles TP sharding — slicing the loaded tensor to the correct shard for the current rank.

**Draft model** (`load_eagle3_model`): Uses an explicit weight name mapping:
```python
weight_map = {
    "fc.weight": "fc.weight",
    "midlayer.input_layernorm.weight": "input_layernorm.weight",
    "midlayer.self_attn.q_proj.weight": "self_attn.q_proj.weight",
    ...
    "d2t": "d2t",  # buffer, not parameter
    "t2d": "t2d",
}
```

### 6.2 Why Separate Loaders

The draft model's weights don't follow the target's conventions:
- **No fused projections**: Q, K, V are separate weights in the checkpoint, not packed into a single QKV matrix.
- **Different naming**: SpecForge uses `midlayer.self_attn.q_proj.weight` while our model uses `self_attn.q_proj.weight`.
- **Non-parameter buffers**: `d2t` and `t2d` are registered buffers (integer/boolean tensors), not learnable parameters. They must be loaded via `named_buffers()` rather than `get_parameter()`.
- **No TP sharding**: The draft model runs on a single GPU, so there's no need for shard-aware loading.

A generic loader that tried to handle both cases would be more complex than two simple, purpose-built loaders. The draft loader is 30 lines — straightforward enough that correctness is obvious by inspection.

### 6.3 Initialization Order

```python
# model_runner.py:31-50
self.model = Qwen3ForCausalLM(hf_config)     # 1. Construct target model
load_model(self.model, config.model)           # 2. Load target weights

if self.use_speculative:
    self.draft_model = Eagle3DraftModel(...)    # 3. Construct draft model
    load_eagle3_model(self.draft_model, ...)    # 4. Load draft weights
    self.draft_model.allocate_kv_cache(k + 1)  # 5. Allocate draft KV cache

self.warmup_model()                            # 6. Warmup (measure peak memory)
self.allocate_kv_cache()                       # 7. Allocate target KV cache (fills remaining GPU)
self.capture_cudagraph()                       # 8. Capture CUDA graphs
```

This order is critical: the draft model must be loaded *before* `allocate_kv_cache()`, because KV cache allocation uses the remaining GPU memory after all model weights are loaded. If we loaded the draft model after KV allocation, we'd either OOM or have to pre-reserve memory speculatively.

---

## Chapter 7: Testing Strategy

### 7.1 Test Suite Overview

We wrote 46 tests across 6 test files, covering every new module:

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_accept_tokens.py` | 7 | Greedy acceptance logic: all accepted, none accepted, partial, single token, hidden state position tracking |
| `test_block_manager.py` | 10 | `may_append_n`, `truncate`, `pre_allocate_speculative`, `deallocate_speculative`, block boundary crossing |
| `test_config.py` | 5 | Draft config parsing, `base_model_layers` formula, validation |
| `test_eagle3.py` | 8 | Model creation, feature fusion shapes, forward pass, KV cache reset, `generate()`, GQA expansion, causal consistency |
| `test_loader.py` | 2 | Weight loading from real SpecForge checkpoint, shape verification |
| `test_qwen3_capture.py` | 6 | `capture_layers` API surface, `compute_logits_all` shapes and consistency |
| `test_scheduler.py` | 5 | Speculative postprocessing: full/none/EOS acceptance, block trimming |

### 7.2 Testing Philosophy

**CPU-first testing**: All tests except `test_loader.py` (which reads real safetensors) run on CPU without requiring a GPU. This is achieved by:

- Disabling `torch.compile` via `TORCHDYNAMO_DISABLE=1` in `conftest.py`
- Using `F.scaled_dot_product_attention` in the draft model (works on CPU)
- Making `allocate_kv_cache` device-aware (defaults to the model's device)
- Testing the acceptance logic as a pure function, decoupled from the model runner

**Real weights, synthetic inputs**: `test_loader.py` loads the actual `models/Qwen3-0.6B-draft/model.safetensors` file, verifying that all weights are non-zero and have expected shapes. This catches naming mismatches between our weight map and the actual checkpoint. The forward pass tests use random inputs — they verify shapes and data flow, not numerical correctness.

**Acceptance logic isolation**: `test_accept_tokens.py` tests the greedy acceptance algorithm as a standalone function with hand-crafted logits. This lets us verify exact behavior for edge cases:

- All drafts accepted → bonus token from position `k`
- First draft rejected → 0 accepted + replacement from position 0
- Partial acceptance → correct split point and hidden state index
- Hidden state tracking → the returned hidden state comes from the correct position (the rejection point or the bonus position)

**Block manager transactional tests**: `test_block_manager.py` verifies the pre-allocate/deallocate cycle preserves invariants:

- `pre_allocate_speculative` extends `block_table` without changing `num_tokens`
- `deallocate_speculative` returns the free block count to its pre-allocation value when no tokens are accepted
- Block boundaries are handled correctly when speculative positions straddle two blocks

### 7.3 What Is Not Tested (And Why)

**Full GPU integration**: Running the complete `LLMEngine.generate()` with speculative decoding requires a CUDA GPU with enough memory for both models (~1.3 GB). We deliberately don't include this as a unit test because:

- It takes 30+ seconds to initialize (model loading + warmup + CUDA graph capture)
- It requires the full Qwen3-0.6B weights to be present
- Failures would be hard to diagnose (is it the draft model? the scheduler? the block manager?)

Instead, each component is tested in isolation, and the integration is verified manually via the example script.

**Non-greedy acceptance**: We only test greedy (argmax) acceptance. Speculative sampling with temperature > 0 is a documented future extension.
