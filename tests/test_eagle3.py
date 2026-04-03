import pytest
import torch
import torch.distributed as dist
from unittest.mock import MagicMock
from types import SimpleNamespace


@pytest.fixture(autouse=True)
def init_dist():
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29501",
                                rank=0, world_size=1)
    yield


class TestEagle3DraftModel:

    def _make_config(self):
        return SimpleNamespace(
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=64,
            intermediate_size=3072,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            draft_vocab_size=32000,
            vocab_size=151936,
        )

    def _make_model(self):
        from nanovllm.models.eagle3 import Eagle3DraftModel
        config = self._make_config()
        with torch.device("cpu"):
            model = Eagle3DraftModel(config, target_hidden_size=1024)
        model.allocate_kv_cache(10)
        return model

    def test_model_creation(self):
        model = self._make_model()
        assert model.fc.weight.shape == (1024, 3072)
        assert model.lm_head.weight.shape == (32000, 1024)
        assert model.d2t.shape == (32000,)
        assert model.t2d.shape == (151936,)

    def test_fuse_features(self):
        model = self._make_model()
        captured = {
            1: torch.randn(1, 1024),
            13: torch.randn(1, 1024),
            24: torch.randn(1, 1024),
        }
        fused = model.fuse_features(captured)
        assert fused.shape == (1, 1024)

    def test_forward_shapes(self):
        model = self._make_model()
        token_embed = torch.randn(1, 1024)
        fused = torch.randn(1, 1024)
        positions = torch.tensor([0])
        logits, hidden = model.forward(token_embed, fused, positions)
        assert logits.shape == (1, 32000)
        assert hidden.shape == (1, 1024)

    def test_kv_cache_reset(self):
        model = self._make_model()
        assert model.self_attn.cache_len == 0
        token_embed = torch.randn(1, 1024)
        fused = torch.randn(1, 1024)
        model.forward(token_embed, fused, torch.tensor([0]))
        assert model.self_attn.cache_len == 1
        model.reset_kv_cache()
        assert model.self_attn.cache_len == 0

    def test_generate_returns_k_tokens(self):
        model = self._make_model()
        model.d2t = torch.arange(32000, dtype=torch.int64)

        def embed_fn(token_ids):
            return torch.randn(token_ids.shape[0], 1024)

        fused = torch.randn(1, 1024)
        start_token = torch.tensor([42])
        draft_tokens = model.generate(embed_fn, fused, start_token, start_pos=10, k=5)
        assert len(draft_tokens) == 5
        assert all(isinstance(t, int) for t in draft_tokens)

    def test_attention_causal_consistency(self):
        model = self._make_model()
        token_embed = torch.randn(1, 1024)
        fused = torch.randn(1, 1024)

        model.reset_kv_cache()
        logits_0, _ = model.forward(token_embed, fused, torch.tensor([0]))
        logits_1, _ = model.forward(token_embed, fused, torch.tensor([1]))

        model.reset_kv_cache()
        logits_0_fresh, _ = model.forward(token_embed, fused, torch.tensor([0]))

        assert torch.allclose(logits_0, logits_0_fresh, atol=1e-5), \
            "First position logits should be identical regardless of subsequent tokens"


class TestEagle3Attention:

    def test_gqa_expansion(self):
        from nanovllm.models.eagle3 import Eagle3Attention
        attn = Eagle3Attention(
            hidden_size=1024, num_heads=16, num_kv_heads=8,
            head_dim=64, max_position=4096, rope_theta=10000.0,
        )
        attn.allocate_kv_cache(10)
        x = torch.randn(1, 2048)
        out = attn(x, torch.tensor([0]))
        assert out.shape == (1, 1024)

    def test_multi_step_kv_accumulation(self):
        from nanovllm.models.eagle3 import Eagle3Attention
        attn = Eagle3Attention(
            hidden_size=1024, num_heads=16, num_kv_heads=8,
            head_dim=64, max_position=4096, rope_theta=10000.0,
        )
        attn.allocate_kv_cache(10)
        for i in range(5):
            x = torch.randn(1, 2048)
            out = attn(x, torch.tensor([i]))
            assert out.shape == (1, 1024)
        assert attn.cache_len == 5
