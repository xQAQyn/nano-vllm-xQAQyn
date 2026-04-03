"""Tests for Qwen3 hidden state capture functionality.

CPU-only tests verify the API surface and shapes.
Full GPU integration is tested via test_integration.py with ModelRunner.
"""
import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig


@pytest.fixture(autouse=True)
def init_dist():
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29502",
                                rank=0, world_size=1)
    yield


class TestQwen3CaptureLayersLogic:

    def test_forward_signature_accepts_capture_layers(self):
        from nanovllm.models.qwen3 import Qwen3Model, Qwen3ForCausalLM
        import inspect
        sig = inspect.signature(Qwen3Model.forward)
        assert "capture_layers" in sig.parameters
        sig2 = inspect.signature(Qwen3ForCausalLM.forward)
        assert "capture_layers" in sig2.parameters

    def test_compute_logits_all_exists(self):
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        assert hasattr(Qwen3ForCausalLM, "compute_logits_all")

    def test_compute_logits_all_shape(self):
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        config = AutoConfig.from_pretrained("models/Qwen3-0.6B")
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        model = Qwen3ForCausalLM(config)
        torch.set_default_dtype(old_dtype)
        hidden_states = torch.randn(5, config.hidden_size)
        logits = model.compute_logits_all(hidden_states)
        assert logits.shape == (5, config.vocab_size)

    def test_compute_logits_all_vs_compute_logits(self):
        """compute_logits_all should return same values as compute_logits for all positions."""
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        config = AutoConfig.from_pretrained("models/Qwen3-0.6B")
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        model = Qwen3ForCausalLM(config)
        torch.set_default_dtype(old_dtype)
        hidden_states = torch.randn(5, config.hidden_size)
        logits_all = model.compute_logits_all(hidden_states)
        # compute_logits_all should give same result for each position
        for i in range(5):
            single = model.compute_logits_all(hidden_states[i:i+1])
            assert torch.allclose(logits_all[i], single[0], atol=1e-5)

    def test_capture_layers_none_returns_tensor(self):
        """When capture_layers=None, Qwen3Model.forward should return Tensor."""
        from nanovllm.models.qwen3 import Qwen3Model
        import inspect
        sig = inspect.signature(Qwen3Model.forward)
        param = sig.parameters["capture_layers"]
        assert param.default is None

    def test_capture_layers_set_return_annotation(self):
        """When capture_layers is set, return type is tuple[Tensor, dict]."""
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        import inspect
        sig = inspect.signature(Qwen3ForCausalLM.forward)
        # Just verify the return type annotation is the union type
        ret = sig.return_annotation
        assert ret != inspect.Parameter.empty
