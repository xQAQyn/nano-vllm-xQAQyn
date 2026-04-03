# Fix: Draft Model KV Cache Persistence Across Speculative Rounds

## Problem

After fixing d2t decoding, residual connection, and fused feedback, EAGLE-3
acceptance was tau=0.65 — still well below training acc0=0.69. Analysis showed
the first-token acceptance rate was only 43.7% (4289 of 7617 rounds rejected
at position 0).

## Root Cause

The draft model's KV cache was **reset at the start of every `generate()`
call**. At step 0 of each speculative round, the self-attention had no prior
context — it attended only to itself, making the attention layer a no-op:

Step 0: Q=K=V from single token → softmax(scalar) = 1.0 → output = V

In SpecForge training, the draft model has full causal attention over the
entire sequence — each position attends to all prior positions. This provides
rich context that the inference code was discarding every round.

**Also confirmed**: the one-step-behind hidden state is NOT a bug. SpecForge
intentionally shifts `input_ids` and `target` but NOT `hidden_states`
(`preprocessing.py:244-245`), so training pairs `hidden(t)` with
`embed(token_{t+1})` — the same lag that occurs at inference.

## Fix: Per-Sequence KV Cache Persistence

### eagle3.py — KV state management

Added `trim_kv_cache`, `save_kv_state`, `restore_kv_state` to both
`Eagle3Attention` and `Eagle3DraftModel`. Removed the `reset_kv_cache()` call
from `generate()` — KV lifecycle is now managed by the caller.

```python
# Eagle3Attention: new methods
def trim_kv_cache(self, keep_len):     # discard rejected entries
def save_kv_state(self):               # clone used portion → (K, V, len)
def restore_kv_state(self, state):     # copy saved state back into cache
model_runner.py — per-sequence orchestration
Added saved_draft_kv: dict[int, tuple] alongside the existing
saved_hidden. The speculative loop now:

Restore the sequence's KV state before generate() (or reset if none exists or cache would overflow the 512-slot buffer)
Generate k draft tokens (KV cache accumulates within the round)
Trim after acceptance: keep prior_len + min(n_accepted + 1, k) entries — all prior context plus the start token and accepted drafts
Save the trimmed KV state back to the per-sequence dict

# Before generate:
state = self.saved_draft_kv.get(seq.seq_id)
if state and state[2] + k <= kv_capacity:
    self.draft_model.restore_kv_state(state)
else:
    self.draft_model.reset_kv_cache()

# After accept_tokens:
self.draft_model.trim_kv_cache(prior_len + min(len(accepted) + 1, k))
self.saved_draft_kv[seq.seq_id] = self.draft_model.save_kv_state()
KV cache allocation increased from k+1 = 6 to 512 slots (~2 MB). On
overflow, the cache resets gracefully. State is cleaned up alongside
saved_hidden when a sequence finishes.

How It Helps
With persistence, step 0 of round N can attend to accepted tokens from all
prior rounds:

Round	Step 0 KV context (before fix)	Step 0 KV context (after fix)
1	0 entries (self only)	0 entries (first round)
2	0 entries (self only)	~2-3 from round 1
5	0 entries (self only)	~8-15 accumulated
20	0 entries (self only)	~40-60 accumulated
The self-attention transitions from a no-op to providing the sequential
context the model was trained to expect.

Memory Cost
Per-sequence overhead: 2 × cache_len × num_kv_heads × head_dim × 2 bytes.
For 100 accumulated entries with 8 KV heads and head_dim=128:
~400 KB per sequence — negligible.

Files Modified
File	Changes
eagle3.py	Added trim_kv_cache, save_kv_state, restore_kv_state to Eagle3Attention and Eagle3DraftModel; removed reset_kv_cache() from generate()
model_runner.py	Added saved_draft_kv dict; restore/trim/save KV state per-sequence in run_speculative(); increased KV allocation to 512; cleanup in clear_saved_hidden()
Tests
All 46 tests pass (2.69s).


# Fix: KV Cache Batch Cross-Contamination

## Bug

When `len(seqs) > 1` in `run_speculative()`, the generate loop processes
sequences serially — each call to `generate()` overwrites the draft model's
KV cache. But the acceptance loop (which runs AFTER all generates) trims and
saves `self.draft_model`'s KV cache for each sequence. At that point, the
cache holds only the LAST sequence's data, so all other sequences get the
wrong KV state saved.

On subsequent rounds, each sequence restores KV entries from an unrelated
sequence, poisoning the self-attention with irrelevant tokens. This explains:

- **8B regression** (tau 0.65 → 0.16): multiple concurrent decode sequences
  trigger the bug, cross-contaminating every sequence's draft context
- **0.6B improvement**: likely only 1 decode sequence at a time (faster
  processing), so the bug never triggers

## Fix

Save each sequence's KV state **immediately after its generate() call** into
a `post_generate_kv` list. In the acceptance loop, trim the saved snapshot
directly instead of touching the live draft model cache:

```python
# Generate loop — snapshot right after each generate
post_generate_kv = []
for seq in seqs:
    ...
    drafts = self.draft_model.generate(...)
    all_draft_tokens.append(drafts)
    post_generate_kv.append(self.draft_model.save_kv_state())  # snapshot here

# Acceptance loop — trim the saved snapshot, not the live cache
for i, seq in enumerate(seqs):
    ...
    kv_state = post_generate_kv[i]        # this seq's own KV data
    k_buf, v_buf, full_len = kv_state
    trim_len = min(keep, full_len)
    self.saved_draft_kv[seq.seq_id] = (k_buf[:trim_len].clone(), ...)
