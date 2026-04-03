import pytest
import torch
import torch.distributed as dist
from types import SimpleNamespace


@pytest.fixture(autouse=True)
def init_dist():
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29503",
                                rank=0, world_size=1)
    yield


class TestEagle3Loader:

    def test_load_eagle3_weights(self):
        from nanovllm.models.eagle3 import Eagle3DraftModel
        from nanovllm.utils.loader import load_eagle3_model

        config = SimpleNamespace(
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
        model = Eagle3DraftModel(config, target_hidden_size=1024)
        load_eagle3_model(model, "models/Qwen3-0.6B-draft")

        # Verify weights are non-zero (loaded successfully)
        assert model.fc.weight.abs().sum() > 0, "fc.weight should be non-zero after loading"
        assert model.lm_head.weight.abs().sum() > 0, "lm_head.weight should be non-zero"
        assert model.self_attn.q_proj.weight.abs().sum() > 0, "q_proj should be non-zero"
        assert model.self_attn.k_proj.weight.abs().sum() > 0, "k_proj should be non-zero"
        assert model.self_attn.v_proj.weight.abs().sum() > 0, "v_proj should be non-zero"
        assert model.self_attn.o_proj.weight.abs().sum() > 0, "o_proj should be non-zero"
        assert model.mlp.gate_proj.weight.abs().sum() > 0, "gate_proj should be non-zero"

        # Verify d2t and t2d buffers
        assert model.d2t.dtype == torch.int64
        assert model.t2d.dtype == torch.bool
        assert model.d2t.shape == (32000,)
        assert model.t2d.shape == (151936,)

    def test_loaded_weight_shapes(self):
        from nanovllm.models.eagle3 import Eagle3DraftModel
        from nanovllm.utils.loader import load_eagle3_model

        config = SimpleNamespace(
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
        model = Eagle3DraftModel(config, target_hidden_size=1024)
        load_eagle3_model(model, "models/Qwen3-0.6B-draft")

        assert model.fc.weight.shape == (1024, 3072)
        assert model.self_attn.q_proj.weight.shape == (1024, 2048)
        assert model.self_attn.k_proj.weight.shape == (512, 2048)
        assert model.self_attn.v_proj.weight.shape == (512, 2048)
        assert model.self_attn.o_proj.weight.shape == (1024, 1024)
        assert model.mlp.gate_proj.weight.shape == (3072, 1024)
        assert model.mlp.up_proj.weight.shape == (3072, 1024)
        assert model.mlp.down_proj.weight.shape == (1024, 3072)
        assert model.lm_head.weight.shape == (32000, 1024)
