# EAGLE-3 Speculative Decoding Debug Report

## Executive Summary

The EAGLE-3 speculative decoding benchmark showed a mean acceptance length (tau) of
**0.00**, with only 5 out of 12,534 speculative steps accepted. Root cause analysis
identified **two bugs** in the nano-vllm inference pipeline — neither related to model
training. After fixing both, all 46 existing unit tests pass.

---

## Benchmark Results (Before Fix)

| Metric                   | Value         |
|--------------------------|---------------|
| Requests                 | 50            |
| Total completion tokens  | 12,589        |
| Throughput               | 156.89 tok/s  |
| Speculative steps        | 12,534        |
| **Mean acceptance (tau)**| **0.00**      |
| **Tree efficiency**      | **0.01%**     |
| 0 accepted               | 12,529        |
| 1 accepted               | 5             |
| 2+ accepted              | 0             |

---

## Investigation Methodology

1. **Traced the full speculative decoding pipeline** from draft generation through
   verification and acceptance, across four key files:
   - `nanovllm/models/eagle3.py` — draft model architecture
   - `nanovllm/engine/model_runner.py` — orchestration + acceptance logic
   - `nanovllm/utils/loader.py` — weight/buffer loading
   - `nanovllm/layers/embed_head.py` — logit computation

2. **Inspected the checkpoint on disk** (`models/Qwen3-0.6B-draft/model.safetensors`)
   to verify all 15 tensors load correctly and examined `d2t`/`t2d` buffer values.

3. **Cross-referenced with the SpecForge training codebase** (`~/SpecForge/`) to
   understand the exact semantics of the `d2t` mapping and the draft model's
   computation graph.

---

## Bug #1 (Primary): d2t Vocabulary Mapping Decoded Incorrectly

### Symptom

The draft model's `d2t` buffer, which maps draft token indices to target token IDs,
appeared to be a many-to-one mapping with only 15,320 unique values for 32,000 entries.
70.7% of the draft vocabulary (22,639 tokens) were unreachable — including extremely
common tokens like `","` (id=11), `"."` (id=13), and `" a"` (id=264).

### Root Cause

SpecForge stores `d2t` with **offset encoding** to compress a monotonically increasing
sequence. In `specforge/data/preprocessing.py:779`:

```python
# SpecForge's generate_vocab_mapping_file:
used_tokens.sort()
d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
```

So `d2t[i]` stores `target_token_id - i`, NOT the target token ID itself. The correct
decoding is:

```
target_token_id = d2t[i] + i
```

But nano-vllm's `eagle3.py:171` used the raw value directly:

```python
target_token = self.d2t[draft_idx]  # WRONG: missing + draft_idx
```

### Verification

```
checkpoint d2t[:30]:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
recovered d2t[:30]:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...]
expected (from t2d):  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...]
Match after fix:      32000 / 32000 (100%)
```

| Token   | Target ID | Draft Index | Checkpoint `d2t[idx]` | Recovered `d2t[idx]+idx` |
|---------|-----------|-------------|-----------------------|--------------------------|
| `" the"`| 279       | 184         | 95                    | 279 ✓                    |
| `"."`   | 13        | 13          | 0                     | 13 ✓                     |
| `","`   | 11        | 11          | 0                     | 11 ✓                     |
| `" a"`  | 264       | 169         | 95                    | 264 ✓                    |
| `"Hello"`| 9707     | 8090        | 1617                  | 9707 ✓                   |

### Fix Applied

**File:** `nanovllm/utils/loader.py` — added offset decoding after buffer loading:

```python
# SpecForge stores d2t with offset encoding: d2t[i] = target_token_id - i
# Decode to direct mapping: d2t[i] = target_token_id
if hasattr(model, "d2t"):
    model.d2t.add_(torch.arange(model.d2t.numel(), device=model.d2t.device))
```

This decodes once at load time so the rest of the codebase can use `d2t` as a simple
direct lookup.

---

## Bug #2 (Secondary): Wrong Residual Connection in Draft Model Forward Pass

### Symptom

Even with correct token mapping, the draft model would produce degraded predictions
because its computation graph diverged from what the weights were trained for.

### Root Cause

SpecForge's `LlamaDecoderLayer` midlayer (`llama3_eagle.py:1285-1307`) uses the
**fc-projected fused features** (pre-norm) as the residual:

```
SpecForge computation graph:
  fc_out = fc(cat(layer1, layer13, layer24))     # project 3072 → 1024
  ┌─ residual = fc_out                            # PRE hidden_norm
  │  normed_fc = hidden_norm(fc_out)
  │  normed_emb = input_layernorm(token_embed)
  │  attn_in = cat(normed_emb, normed_fc)
  │  attn_out = self_attn(attn_in)
  └─ hidden = fc_out + attn_out                   # ← residual from fc_out
     ...MLP with its own residual...
```

nano-vllm's original code had two differences:

1. `fuse_features()` applied `hidden_norm` eagerly, returning the normed output.
2. `forward()` used `token_embed` (not `fc_out`) as the attention residual.

```python
# BEFORE (wrong):
def fuse_features(self, captured):
    return self.hidden_norm(self.fc(cat_features))  # returns normed

def forward(self, token_embed, fused, positions):
    normed = self.input_layernorm(token_embed)
    attn_input = torch.cat([normed, fused], dim=-1)  # fused already normed ✓
    attn_out = self.self_attn(attn_input, positions)
    hidden = token_embed + attn_out                   # ✗ wrong residual
```

The attention inputs happened to be equivalent (both `[normed_embed, normed_fc_out]`),
but the residual was `token_embed` instead of `fc_out`.

### Fix Applied

**File:** `nanovllm/models/eagle3.py` — moved `hidden_norm` into `forward()` and
corrected the residual:

```python
# AFTER (correct):
def fuse_features(self, captured):
    return self.fc(cat_features)           # return PRE-norm

def forward(self, token_embed, fused, positions):
    normed_fused = self.hidden_norm(fused)              # norm here instead
    normed_embed = self.input_layernorm(token_embed)
    attn_input = torch.cat([normed_embed, normed_fused], dim=-1)
    attn_out = self.self_attn(attn_input, positions)
    hidden = fused + attn_out                           # ✓ correct residual
```

---

## Hypotheses Investigated and Ruled Out

| Hypothesis                              | Status   | Evidence |
|-----------------------------------------|----------|----------|
| `accept_tokens` off-by-one indexing     | Ruled out| `seq_logits[i]` predicts position P+i; `draft_tokens[i]` predicts position P+i. Verified by tracing `prepare_verify` which builds `[last_token] + drafts` at positions `[P-1, P, ..., P+k-1]`, so `logits[0]` predicts P matching `draft[0]`. |
| Weights silently not loaded             | Ruled out| All 15 checkpoint keys have entries in `weight_map`. Verified programmatically. |
| `t2d` buffer corrupted                  | Ruled out| Correctly loaded: exactly 32,000 True values spanning target token IDs [0, 151668]. |
| Feature fusion layer order mismatch     | Ruled out| Code uses `sorted(captured)` → ascending [1, 13, 24]. SpecForge concatenates low→mid→high in the same order. |
| Draft model uses wrong architecture for Qwen3 | Ruled out| SpecForge trains Qwen3 using `LlamaForCausalLMEagle3` class. The draft config `architectures: ["LlamaForCausalLMEagle3"]` is correct — it's a generic single-layer decoder, not Llama-specific. |

---

## Test Results After Fix

```
46 passed in 2.79s
```

All existing tests pass, covering:
- Acceptance logic (6 tests)
- Block manager speculative protocol (8 tests)
- Config validation (5 tests)
- Eagle3 draft model architecture (8 tests)
- Weight loader (2 tests)
- Qwen3 capture layers (6 tests)
- Scheduler speculative postprocessing (5 tests)
- Misc (6 tests)

---

## Files Modified

| File | Change |
|------|--------|
| `nanovllm/utils/loader.py:64-67` | Added d2t offset decoding after buffer load |
| `nanovllm/models/eagle3.py:134-153` | Moved `hidden_norm` from `fuse_features` to `forward`; changed residual from `token_embed` to `fused` |

---

## Recommended Next Steps

1. **Re-run benchmark on GPU machine** — expect tau >> 1.0 and tree efficiency > 50%.
2. **Correctness check** — run identical prompts with and without speculative decoding
   at temperature=0; output tokens should match exactly.
3. **Consider documenting the SpecForge d2t convention** in the codebase to prevent
   future confusion.
