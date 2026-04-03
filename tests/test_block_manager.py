import pytest
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class TestBlockManagerAppendN:

    def setup_method(self):
        self.block_size = 256
        self.bm = BlockManager(num_blocks=100, block_size=self.block_size)

    def _make_seq(self, n_tokens):
        seq = Sequence([0] * n_tokens)
        self.bm.allocate(seq)
        return seq

    def test_can_append_n_no_new_blocks(self):
        seq = self._make_seq(10)
        assert self.bm.can_append_n(seq, 5)

    def test_can_append_n_needs_new_block(self):
        seq = self._make_seq(self.block_size)
        assert self.bm.can_append_n(seq, 1)

    def test_may_append_n_single(self):
        seq = self._make_seq(10)
        old_blocks = len(seq.block_table)
        seq.append_token(99)
        self.bm.may_append_n(seq, 1)
        assert len(seq.block_table) >= old_blocks

    def test_may_append_n_multiple(self):
        seq = self._make_seq(self.block_size - 1)
        seq.append_token(1)
        self.bm.may_append(seq)
        for i in range(5):
            seq.append_token(i + 10)
            self.bm.may_append(seq)
        assert seq.num_tokens == self.block_size + 5

    def test_may_append_n_crosses_block_boundary(self):
        seq = self._make_seq(self.block_size - 2)
        initial_blocks = len(seq.block_table)
        for i in range(5):
            seq.append_token(i)
            self.bm.may_append(seq)
        assert len(seq.block_table) == initial_blocks + 1


class TestBlockManagerSpeculative:

    def setup_method(self):
        self.block_size = 256
        self.bm = BlockManager(num_blocks=100, block_size=self.block_size)

    def _make_seq(self, n_tokens):
        seq = Sequence([0] * n_tokens)
        self.bm.allocate(seq)
        return seq

    def test_pre_allocate_no_new_blocks_needed(self):
        seq = self._make_seq(10)
        initial_blocks = len(seq.block_table)
        new_blocks = self.bm.pre_allocate_speculative(seq, 5)
        # 10 + 5 = 15, still within first block (256)
        assert new_blocks == 0
        assert len(seq.block_table) == initial_blocks

    def test_pre_allocate_crosses_boundary(self):
        seq = self._make_seq(self.block_size - 2)
        initial_blocks = len(seq.block_table)
        free_before = len(self.bm.free_block_ids)
        # Need 5 more positions: some cross into next block
        new_blocks = self.bm.pre_allocate_speculative(seq, 5)
        assert new_blocks == 1
        assert len(seq.block_table) == initial_blocks + 1
        assert len(self.bm.free_block_ids) == free_before - 1

    def test_pre_allocate_does_not_change_seq_tokens(self):
        seq = self._make_seq(10)
        old_len = seq.num_tokens
        self.bm.pre_allocate_speculative(seq, 5)
        assert seq.num_tokens == old_len
        assert len(seq.token_ids) == old_len

    def test_deallocate_speculative_trims_extra_blocks(self):
        seq = self._make_seq(self.block_size - 2)
        initial_blocks = len(seq.block_table)
        free_before = len(self.bm.free_block_ids)
        self.bm.pre_allocate_speculative(seq, 5)
        assert len(seq.block_table) == initial_blocks + 1
        # Accept 0 tokens — no tokens appended, trim back
        self.bm.deallocate_speculative(seq, 0)
        assert len(seq.block_table) == initial_blocks
        assert len(self.bm.free_block_ids) == free_before

    def test_deallocate_speculative_keeps_needed_blocks(self):
        seq = self._make_seq(self.block_size - 1)
        self.bm.pre_allocate_speculative(seq, 5)
        # Simulate accepting 3 tokens
        for i in range(3):
            seq.append_token(i + 100)
        # Now seq.num_tokens = block_size + 2, needs 2 blocks
        self.bm.deallocate_speculative(seq, 3)
        assert len(seq.block_table) == 2  # both needed


class TestBlockManagerTruncate:

    def setup_method(self):
        self.block_size = 256
        self.bm = BlockManager(num_blocks=100, block_size=self.block_size)

    def _make_seq(self, n_tokens):
        seq = Sequence([0] * n_tokens)
        self.bm.allocate(seq)
        return seq

    def test_truncate_within_same_block(self):
        seq = self._make_seq(10)
        old_blocks = len(seq.block_table)
        self.bm.truncate(seq, 5)
        assert seq.num_tokens == 5
        assert len(seq.block_table) == old_blocks

    def test_truncate_removes_blocks(self):
        seq = self._make_seq(self.block_size + 10)
        free_before = len(self.bm.free_block_ids)
        assert len(seq.block_table) == 2
        self.bm.truncate(seq, self.block_size - 5)
        assert len(seq.block_table) == 1
        assert seq.num_tokens == self.block_size - 5
        assert len(self.bm.free_block_ids) == free_before + 1

    def test_truncate_preserves_last_token(self):
        tokens = list(range(20))
        seq = Sequence(tokens)
        self.bm.allocate(seq)
        self.bm.truncate(seq, 10)
        assert seq.last_token == 9
        assert seq.token_ids == list(range(10))
