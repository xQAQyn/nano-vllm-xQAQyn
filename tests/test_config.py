import pytest
from nanovllm.config import Config


class TestConfig:

    def test_base_config(self):
        config = Config(model="models/Qwen3-0.6B")
        assert config.hf_config is not None
        assert config.hf_config.num_hidden_layers == 28
        assert config.draft_model is None
        assert config.base_model_layers is None
        assert config.draft_hf_config is None

    def test_speculative_config(self):
        config = Config(model="models/Qwen3-0.6B", draft_model="models/Qwen3-0.6B-draft")
        assert config.draft_hf_config is not None
        assert config.base_model_layers == [1, 13, 24]
        assert config.num_speculative_tokens == 5
        assert config.draft_hf_config.num_hidden_layers == 1

    def test_base_model_layers_formula(self):
        config = Config(model="models/Qwen3-0.6B", draft_model="models/Qwen3-0.6B-draft")
        n = config.hf_config.num_hidden_layers  # 28
        assert config.base_model_layers == [1, n // 2 - 1, n - 4]

    def test_custom_speculative_tokens(self):
        config = Config(model="models/Qwen3-0.6B", draft_model="models/Qwen3-0.6B-draft",
                        num_speculative_tokens=3)
        assert config.num_speculative_tokens == 3

    def test_invalid_draft_model_path(self):
        with pytest.raises(AssertionError):
            Config(model="models/Qwen3-0.6B", draft_model="models/nonexistent")
