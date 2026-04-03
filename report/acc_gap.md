# EAGLE-3 Acceptance Gap Analysis: Training acc vs Inference tau

## Problem Statement

After fixing the d2t decoding and residual connection bugs, the benchmark
acceptance improved from 0.00 to **tau = 0.58**. However, the SpecForge
training checkpoint reports **acc0 = 0.69** declining to **acc6 = 0.55**,
suggesting the inference acceptance should be much higher. This report
explains the gap and identifies the remaining bug.

---

## 1. What SpecForge acc0–acc6 Actually Measure

SpecForge trains with **TTT (Test-Time Training)**, an unrolled autoregressive
loop of `length=7` steps (idx 0–6). Each step:

```python
Step 0:
hidden_states = fc(target_model_hidden)       # per-position, [batch, seq_len, hidden]
input_embeds  = embed(ground_truth_input_ids)  # per-position, [batch, seq_len, hidden]
hidden_out    = backbone(input_embeds, hidden_states)  # full causal attention
logits        = lm_head(norm(hidden_out))
acc0 = (logits.argmax == target_p.argmax) averaged over all positions

Step 1:
hidden_states = hidden_out  (from step 0)      # draft's own output, per-position
input_ids     = shift_left(input_ids)           # [t1, t2, ..., tn, 0]
input_embeds  = embed(shifted_input_ids)
hidden_out    = backbone(input_embeds, hidden_states)
logits        = lm_head(norm(hidden_out))
acc1 = argmax match ...
```

Step 2–6: same pattern, hidden_states always = previous step's output



**Key properties of training evaluation:**
- Step 0 uses **target model hidden states at every position** — the richest possible context
- Steps 1–6 use the **draft model's own previous output** as hidden states
- At every step, the draft model processes **the entire sequence in parallel** with causal attention — each position t has its own hidden state from the full context up to t
- `position_mask` excludes positions where the target's argmax falls outside the draft vocabulary

**acc0 = 0.69** means: with perfect per-position hidden states from the target model, the draft model's argmax matches the target's argmax at 69% of positions.

**acc6 = 0.55** means: after 6 rounds of using its own outputs, accuracy degrades to 55%.

---

## 2. How Inference Differs

In `eagle3.py` `generate()`, the draft model runs k=5 steps autoregressively:

```python
def generate(self, embed_fn, fused, start_token, start_pos, k):
    self.reset_kv_cache()
    token = start_token
    for i in range(k):
        token_embed = embed_fn(token)
        positions = torch.tensor([start_pos + i], ...)
        logits, _ = self.forward(token_embed, fused, positions)  # fused is STATIC
        #       ^^^ hidden output DISCARDED
        draft_idx = logits.argmax(dim=-1)
        target_token = self.d2t[draft_idx]
        token = target_token
    return draft_tokens
```

Difference 1 (CRITICAL BUG): fused is never updated between steps
In training, the TTT loop feeds back hidden_states_out as the next step's
hidden_states:

```python
# SpecForge TTT loop (eagle3.py:249-250)
hidden_states_out = self.draft_model.backbone(input_embeds, state.hidden_states, ...)
hidden_states = hidden_states_out  # ← UPDATES for next step
```
But in nano-vllm's generate(), fused (the equivalent of hidden_states)
is computed ONCE before the loop and never updated:

```python
# model_runner.py:305
fused = self.draft_model.fuse_features(self.saved_hidden[seq.seq_id])
drafts = self.draft_model.generate(embed_fn, fused, ...)  # fused stays static
```
Inside generate(), self.forward() returns (logits, hidden) but the
hidden output is discarded (logits, _ = ...). The model uses the same
stale fused for all k steps.

Impact: The draft model cannot condition on its own evolving representations.
In training, steps 1–6 see the draft's own output (which still has per-position
context). At inference, all 5 steps see the identical frozen representation from
one target model position. This is the dominant cause of the accuracy gap.

Difference 2: Single position vs full sequence
Aspect	Training	Inference
Hidden states	Per-position across full sequence	ONE vector from ONE position
Attention	Full causal over seq_len positions	Builds KV cache from 1-token steps
Context at pos t	All prior positions' hidden states	Only prior draft steps' KV entries
In training, even at step 1+, the draft model has per-position hidden
states for the entire sequence, because the backbone runs full causal attention
over all positions simultaneously. At inference, the draft model processes one
token at a time, accumulating context only through its own KV cache.

Difference 3: Cross-round hidden state staleness
After a verification round where 0 tokens are accepted (the most common
case with low tau):


# accept_tokens (model_runner.py:290-292)
accept_pos = 0  # first position of verify input
new_hidden = {l: seq_captured[l][0:1] for l in seq_captured}
The saved_hidden is set to the target model's hidden state from processing
last_token at position L-1 — one position before the newly committed
token. The next speculative round's fused therefore doesn't reflect the most
recently committed token at position L.

When tokens are accepted (n > 0), saved_hidden comes from position
L-1+n which includes correct context. So this issue creates a negative feedback
loop: low acceptance → stale hidden → lower acceptance → ...

3. Expected vs Observed Tau
Given these mismatches, we can reason about the expected tau:

Training acc0 = 0.69 with perfect per-position target model hidden states
At inference, the draft model has one stale hidden state for all positions
Rough estimate: per-step acceptance ≈ 0.35–0.40 (significantly below 0.69)
With p ≈ 0.37 (independent model), expected tau ≈ 0.37 + 0.14 + 0.05 + 0.02 + 0.01 ≈ 0.59
This closely matches the observed tau = 0.58.