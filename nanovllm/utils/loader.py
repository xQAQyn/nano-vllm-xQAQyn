import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_eagle3_model(model: nn.Module, path: str):
    weight_map = {
        "fc.weight": "fc.weight",
        "hidden_norm.weight": "hidden_norm.weight",  # mapped from midlayer.hidden_norm
        "midlayer.hidden_norm.weight": "hidden_norm.weight",
        "midlayer.input_layernorm.weight": "input_layernorm.weight",
        "midlayer.self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "midlayer.self_attn.k_proj.weight": "self_attn.k_proj.weight",
        "midlayer.self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "midlayer.self_attn.o_proj.weight": "self_attn.o_proj.weight",
        "midlayer.post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "midlayer.mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "midlayer.mlp.up_proj.weight": "mlp.up_proj.weight",
        "midlayer.mlp.down_proj.weight": "mlp.down_proj.weight",
        "norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
        "d2t": "d2t",
        "t2d": "t2d",
    }
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                param_name = weight_map.get(weight_name)
                if param_name is None:
                    continue
                tensor = f.get_tensor(weight_name)
                param = dict(model.named_parameters()).get(param_name)
                if param is None:
                    buf = dict(model.named_buffers()).get(param_name)
                    if buf is not None:
                        buf.copy_(tensor)
                    continue
                param.data.copy_(tensor)
    # SpecForge stores d2t with offset encoding: d2t[i] = target_token_id - i
    # Decode to direct mapping: d2t[i] = target_token_id
    if hasattr(model, "d2t"):
        model.d2t.add_(torch.arange(model.d2t.numel(), device=model.d2t.device))
