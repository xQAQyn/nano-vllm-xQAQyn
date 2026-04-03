import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb


class Eagle3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        input_size = hidden_size * 2

        self.q_proj = nn.Linear(input_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(input_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(head_dim, head_dim, max_position, rope_theta)
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        self.cache_len = 0

    def allocate_kv_cache(self, max_len: int, device=None, dtype=None):
        device = device or self.q_proj.weight.device
        dtype = dtype or self.q_proj.weight.dtype
        self.k_cache = torch.zeros(max_len, self.num_kv_heads, self.head_dim,
                                   device=device, dtype=dtype)
        self.v_cache = torch.zeros(max_len, self.num_kv_heads, self.head_dim,
                                   device=device, dtype=dtype)
        self.cache_len = 0

    def reset_kv_cache(self):
        self.cache_len = 0

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)

        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        q = apply_rotary_emb(q, cos, sin).to(k.dtype)
        k = apply_rotary_emb(k, cos, sin).to(v.dtype)

        n = k.size(0)
        self.k_cache[self.cache_len:self.cache_len + n] = k
        self.v_cache[self.cache_len:self.cache_len + n] = v
        self.cache_len += n

        k_full = self.k_cache[:self.cache_len].to(q.dtype)
        v_full = self.v_cache[:self.cache_len].to(q.dtype)

        q = q.unsqueeze(0).transpose(1, 2)
        k_full = k_full.unsqueeze(0).transpose(1, 2)
        v_full = v_full.unsqueeze(0).transpose(1, 2)

        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k_full = k_full.repeat_interleave(repeats, dim=1)
            v_full = v_full.repeat_interleave(repeats, dim=1)

        o = F.scaled_dot_product_attention(q, k_full, v_full,
                                           is_causal=(n > 1),
                                           scale=self.scaling)
        o = o.transpose(1, 2).reshape(-1, self.num_heads * self.head_dim)
        return self.o_proj(o)


class Eagle3MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3DraftModel(nn.Module):

    def __init__(self, config, target_hidden_size: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        num_layers_fused = 3

        self.fc = nn.Linear(target_hidden_size * num_layers_fused, hidden_size, bias=False)
        self.hidden_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Eagle3Attention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, 'head_dim', hidden_size // config.num_attention_heads),
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, 'rope_theta', 10000.0),
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.mlp = Eagle3MLP(hidden_size, config.intermediate_size)
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        draft_vocab_size = getattr(config, 'draft_vocab_size', config.vocab_size)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)

        self.register_buffer("d2t", torch.zeros(draft_vocab_size, dtype=torch.int64))
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))

    def allocate_kv_cache(self, max_len: int):
        self.self_attn.allocate_kv_cache(max_len)

    def reset_kv_cache(self):
        self.self_attn.reset_kv_cache()

    def fuse_features(self, captured: dict[int, torch.Tensor]) -> torch.Tensor:
        cat_features = torch.cat([captured[l] for l in sorted(captured)], dim=-1)
        return self.fc(cat_features)

    def forward(
        self,
        token_embed: torch.Tensor,
        fused: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed_fused = self.hidden_norm(fused)
        normed_embed = self.input_layernorm(token_embed)
        attn_input = torch.cat([normed_embed, normed_fused], dim=-1)
        attn_out = self.self_attn(attn_input, positions)
        hidden = fused + attn_out
        normed = self.post_attention_layernorm(hidden)
        mlp_out = self.mlp(normed)
        hidden = hidden + mlp_out
        logits = self.lm_head(self.norm(hidden))
        return logits, hidden

    @torch.inference_mode()
    def generate(
        self,
        embed_fn,
        fused: torch.Tensor,
        start_token: torch.Tensor,
        start_pos: int,
        k: int,
    ) -> list[int]:
        self.reset_kv_cache()
        token = start_token
        draft_tokens = []
        for i in range(k):
            token_embed = embed_fn(token)
            positions = torch.tensor([start_pos + i], device=token.device, dtype=torch.long)
            logits, fused = self.forward(token_embed, fused, positions)
            draft_idx = logits.argmax(dim=-1)
            target_token = self.d2t[draft_idx]
            draft_tokens.append(target_token.item())
            token = target_token
        return draft_tokens
