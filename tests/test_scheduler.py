import pytest
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.config import Config


class TestSchedulerSpeculative:

    def setup_method(self):
        self.config = Config(model="models/Qwen3-0.6B")
        self.config.eos = 151645
        self.config.num_kvcache_blocks = 100

    def _make_scheduler(self):
        return Scheduler(self.config)

    def _prefill_seq(self, scheduler, token_ids):
        seq = Sequence(token_ids)
        scheduler.add(seq)
        seqs, is_prefill = scheduler.schedule()
        assert is_prefill
        scheduler.postprocess(seqs, [10])
        return seq

    def test_postprocess_speculative_all_accepted(self):
        scheduler = self._make_scheduler()
        seq = self._prefill_seq(scheduler, [1, 2, 3])

        seqs, is_prefill = scheduler.schedule()
        assert not is_prefill

        # Pre-allocate like the real flow would
        scheduler.block_manager.pre_allocate_speculative(seqs[0], 5)

        results = [([20, 30, 40], 50)]  # 3 accepted + 1 bonus
        scheduler.postprocess_speculative(seqs, results)

        assert seq.num_tokens == 4 + 4  # 3 original + 1 from prefill + 4 from speculative
        assert seq.token_ids[-4:] == [20, 30, 40, 50]

    def test_postprocess_speculative_none_accepted(self):
        scheduler = self._make_scheduler()
        seq = self._prefill_seq(scheduler, [1, 2, 3])

        seqs, _ = scheduler.schedule()
        scheduler.block_manager.pre_allocate_speculative(seqs[0], 2)

        results = [([], 99)]  # 0 accepted + 1 replacement
        scheduler.postprocess_speculative(seqs, results)

        assert seq.token_ids[-1] == 99
        assert seq.num_tokens == 5  # 3 + 1 + 1

    def test_postprocess_speculative_eos_in_accepted(self):
        scheduler = self._make_scheduler()
        seq = self._prefill_seq(scheduler, [1, 2, 3])

        seqs, _ = scheduler.schedule()
        scheduler.block_manager.pre_allocate_speculative(seqs[0], 5)

        eos = self.config.eos
        # EOS is the second accepted token
        results = [([20, eos, 40], 50)]
        scheduler.postprocess_speculative(seqs, results)

        assert seq.is_finished
        # Should stop at EOS, not append remaining tokens
        assert seq.token_ids[-1] == eos

    def test_postprocess_speculative_eos_as_new_token(self):
        scheduler = self._make_scheduler()
        seq = self._prefill_seq(scheduler, [1, 2, 3])

        seqs, _ = scheduler.schedule()
        scheduler.block_manager.pre_allocate_speculative(seqs[0], 4)

        eos = self.config.eos
        results = [([20, 30], eos)]
        scheduler.postprocess_speculative(seqs, results)

        assert seq.is_finished
        assert seq.token_ids[-1] == eos

    def test_speculative_blocks_trimmed_on_partial_accept(self):
        scheduler = self._make_scheduler()
        seq = self._prefill_seq(scheduler, [0] * 254)

        seqs, _ = scheduler.schedule()
        # seq is now 255 tokens. Pre-allocate for 5 speculative tokens.
        # Positions 255-259 need a second block (255 is at block_idx=0, 256 starts block_idx=1)
        free_before = len(scheduler.block_manager.free_block_ids)
        scheduler.block_manager.pre_allocate_speculative(seqs[0], 5)
        assert len(scheduler.block_manager.free_block_ids) < free_before

        # Accept only 0 tokens + 1 bonus = total 1 token appended
        results = [([], 99)]
        scheduler.postprocess_speculative(seqs, results)

        # seq is now 256 tokens = 1 block exactly, no extra block needed
        assert seq.num_tokens == 256
        # The extra block should have been deallocated
        assert len(scheduler.block_manager.free_block_ids) == free_before
