import pytest
import torch


class TestAcceptTokens:
    """Test the greedy acceptance logic in isolation."""

    def _accept_tokens(self, seq_logits, draft_tokens, seq_captured):
        """Standalone version of ModelRunner.accept_tokens for unit testing."""
        target_tokens = seq_logits.argmax(dim=-1)
        accepted = []
        for i, dt in enumerate(draft_tokens):
            if target_tokens[i].item() == dt:
                accepted.append(dt)
            else:
                new_token = target_tokens[i].item()
                accept_pos = i
                new_hidden = {l: seq_captured[l][accept_pos:accept_pos+1] for l in seq_captured}
                return accepted, new_token, new_hidden
        new_token = target_tokens[len(draft_tokens)].item()
        accept_pos = len(draft_tokens)
        new_hidden = {l: seq_captured[l][accept_pos:accept_pos+1] for l in seq_captured}
        return accepted, new_token, new_hidden

    def test_all_accepted(self):
        vocab_size = 100
        draft_tokens = [10, 20, 30]
        # logits[0] argmax=10, logits[1] argmax=20, logits[2] argmax=30, logits[3] argmax=50
        logits = torch.zeros(4, vocab_size)
        logits[0, 10] = 100.0
        logits[1, 20] = 100.0
        logits[2, 30] = 100.0
        logits[3, 50] = 100.0  # bonus token
        captured = {1: torch.randn(4, 1024), 13: torch.randn(4, 1024)}
        accepted, new_token, new_hidden = self._accept_tokens(logits, draft_tokens, captured)
        assert accepted == [10, 20, 30]
        assert new_token == 50
        assert new_hidden[1].shape == (1, 1024)

    def test_none_accepted(self):
        vocab_size = 100
        draft_tokens = [10, 20, 30]
        logits = torch.zeros(4, vocab_size)
        logits[0, 99] = 100.0  # mismatch at first position
        logits[1, 20] = 100.0
        logits[2, 30] = 100.0
        logits[3, 50] = 100.0
        captured = {1: torch.randn(4, 1024)}
        accepted, new_token, new_hidden = self._accept_tokens(logits, draft_tokens, captured)
        assert accepted == []
        assert new_token == 99

    def test_partial_acceptance(self):
        vocab_size = 100
        draft_tokens = [10, 20, 30, 40, 50]
        logits = torch.zeros(6, vocab_size)
        logits[0, 10] = 100.0
        logits[1, 20] = 100.0
        logits[2, 77] = 100.0  # mismatch at position 2
        logits[3, 40] = 100.0
        logits[4, 50] = 100.0
        logits[5, 60] = 100.0
        captured = {1: torch.randn(6, 1024)}
        accepted, new_token, new_hidden = self._accept_tokens(logits, draft_tokens, captured)
        assert accepted == [10, 20]
        assert new_token == 77

    def test_single_draft_accepted(self):
        vocab_size = 100
        draft_tokens = [42]
        logits = torch.zeros(2, vocab_size)
        logits[0, 42] = 100.0
        logits[1, 7] = 100.0
        captured = {1: torch.randn(2, 1024)}
        accepted, new_token, _ = self._accept_tokens(logits, draft_tokens, captured)
        assert accepted == [42]
        assert new_token == 7

    def test_single_draft_rejected(self):
        vocab_size = 100
        draft_tokens = [42]
        logits = torch.zeros(2, vocab_size)
        logits[0, 99] = 100.0
        logits[1, 7] = 100.0
        captured = {1: torch.randn(2, 1024)}
        accepted, new_token, _ = self._accept_tokens(logits, draft_tokens, captured)
        assert accepted == []
        assert new_token == 99

    def test_hidden_state_at_correct_position(self):
        vocab_size = 100
        draft_tokens = [10, 20, 30]
        logits = torch.zeros(4, vocab_size)
        logits[0, 10] = 100.0
        logits[1, 99] = 100.0  # reject at position 1
        logits[2, 30] = 100.0
        logits[3, 50] = 100.0
        captured_tensor = torch.arange(4).unsqueeze(1).expand(4, 1024).float()
        captured = {1: captured_tensor}
        accepted, new_token, new_hidden = self._accept_tokens(logits, draft_tokens, captured)
        assert accepted == [10]
        assert new_token == 99
        # hidden state should be from position 1 (the rejection position)
        assert new_hidden[1][0, 0].item() == 1.0

    def test_all_accepted_hidden_at_last(self):
        vocab_size = 100
        draft_tokens = [10, 20]
        logits = torch.zeros(3, vocab_size)
        logits[0, 10] = 100.0
        logits[1, 20] = 100.0
        logits[2, 50] = 100.0
        captured_tensor = torch.arange(3).unsqueeze(1).expand(3, 1024).float()
        captured = {1: captured_tensor}
        accepted, new_token, new_hidden = self._accept_tokens(logits, draft_tokens, captured)
        assert accepted == [10, 20]
        assert new_token == 50
        # hidden state should be from position 2 (bonus token position)
        assert new_hidden[1][0, 0].item() == 2.0
