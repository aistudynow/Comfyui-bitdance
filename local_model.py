from __future__ import annotations
# Created by aistudynow.com

import copy
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore
try:
    from transformers import Qwen3Config  # type: ignore
except Exception:  # pragma: no cover
    Qwen3Config = None
from transformers.activations import ACT2FN  # type: ignore

# NOTE:
# This module intentionally binds to the official BitDance architecture files.
# If you move this node pack out of the BitDance repository, copy the official
# model classes into this file (or alongside it) and keep these function names.
from .bitdance_arch.vision_encoder.autoencoder import VQModel  # type: ignore
from .bitdance_arch.vision_head.flow_head_parallel_x import DiffHead, set_bitdance_attention_mode  # type: ignore


class MLPconnector(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = torch.nn.Linear(in_dim, out_dim)
        self.fc2 = torch.nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


DEFAULT_BITDANCE_64X_DIFFUSERS_AUTOENCODER_CONFIG: Dict[str, Any] = {
    "gan_decoder": False,
    "ddconfig": {
        "ch": 256,
        "ch_mult": [1, 1, 2, 2, 4],
        "double_z": False,
        "in_channels": 3,
        "num_res_blocks": 4,
        "out_ch": 3,
        "z_channels": 32,
    },
}

DEFAULT_BITDANCE_64X_DIFFUSERS_DIFFUSION_HEAD_CONFIG: Dict[str, Any] = {
    "P_mean": 0.0,
    "P_std": 1.0,
    "ch_cond": 5120,
    "ch_latent": 5120,
    "ch_target": 32,
    "depth_adanln": 2,
    "depth_latent": 6,
    "diff_batch_mul": 1,
    "grad_checkpointing": False,
    "parallel_num": 64,
    "time_schedule": "logit_normal",
    "time_shift": 1.0,
    "use_swiglu": True,
}

DEFAULT_BITDANCE_64X_DIFFUSERS_PROJECTOR_CONFIG: Dict[str, Any] = {
    "hidden_act": "gelu_pytorch_tanh",
    "in_dim": 32,
    "out_dim": 5120,
}

DEFAULT_BITDANCE_64X_DIFFUSERS_TEXT_ENCODER_CONFIG: Dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "dtype": "bfloat16",
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 17408,
    "layer_types": ["full_attention"] * 40,
    "max_position_embeddings": 40960,
    "max_window_layers": 40,
    "model_type": "qwen3",
    "num_attention_heads": 40,
    "num_hidden_layers": 40,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "transformers_version": "4.57.3",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151988,
}


def strip_diffusers_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in config.items() if not str(k).startswith("_")}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_hidden_size(model_root: Path) -> int:
    config = _read_json(model_root / "config.json")
    hidden_size = config.get("hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Missing hidden_size in {model_root / 'config.json'}")
    return int(hidden_size)


def set_vision_attention_mode(mode: str = "auto") -> None:
    mode = str(mode or "auto")
    if mode == "eager":
        # Our vision head "eager" path is the non-flash/non-sdpa fallback.
        set_bitdance_attention_mode("eager")
        return
    if mode == "sdpa":
        set_bitdance_attention_mode("sdpa")
        return
    if mode in ("flash_attn_2", "flash_attn_3"):
        set_bitdance_attention_mode(mode)
        return
    set_bitdance_attention_mode("auto")


def _normalize_text_attention_mode(mode: str) -> str | None:
    mode = str(mode or "auto")
    if mode in ("auto", "default"):
        return None
    if mode == "flash_attn_2":
        return "flash_attention_2"
    if mode == "sdpa":
        return "sdpa"
    if mode == "eager":
        return "eager"
    # flash_attn_3 / other custom modes not directly supported by HF Qwen config
    return None


def _replace_rmsnorm_with_pytorch(module: nn.Module) -> int:
    if not hasattr(nn, "RMSNorm"):
        return 0
    replaced = 0
    for name, child in list(module.named_children()):
        child_replaced = 0
        cls_name = child.__class__.__name__.lower()
        if "rmsnorm" in cls_name and hasattr(child, "weight"):
            weight = getattr(child, "weight", None)
            if isinstance(weight, torch.Tensor):
                eps = getattr(child, "variance_epsilon", getattr(child, "eps", 1e-6))
                new_mod = nn.RMSNorm(
                    weight.shape,  # type: ignore
                    eps=float(eps),
                    elementwise_affine=True,
                    device=weight.device,  # type: ignore
                    dtype=weight.dtype,  # type: ignore
                )
                if weight.device.type != "meta":  # type: ignore
                    with torch.no_grad():
                        new_mod.weight.copy_(weight)
                setattr(module, name, new_mod)
                replaced += 1
                child_replaced = 1
        if child_replaced == 0:
            replaced += _replace_rmsnorm_with_pytorch(child)
    return replaced


def _configure_text_model_runtime(
    model: torch.nn.Module,
    config: Any,
    *,
    attention_mode: str = "auto",
    rms_norm_function: str = "default",
) -> None:
    attn_impl = _normalize_text_attention_mode(attention_mode)
    if attn_impl is not None:
        try:
            setattr(config, "_attn_implementation", attn_impl)
        except Exception:
            pass
        try:
            if hasattr(model, "config"):
                setattr(model.config, "_attn_implementation", attn_impl)
        except Exception:
            pass
    if str(rms_norm_function or "default") == "pytorch":
        _replace_rmsnorm_with_pytorch(model)


def build_vision_head(model_root: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    config = _read_json(model_root / "vision_head_config.json")
    model = DiffHead(**config).eval()
    return model, config


def build_vision_head_from_config(config: Dict[str, Any]) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    cfg = strip_diffusers_metadata(copy.deepcopy(config))
    model = DiffHead(**cfg).eval()
    return model, cfg


def build_vae(model_root: Path) -> Tuple[torch.nn.Module, Dict[str, Any], int]:
    config = _read_json(model_root / "ae_config.json")
    model = VQModel(**config).eval()
    vae_patch_size = 2 ** (len(config["ddconfig"]["ch_mult"]) - 1)
    return model, config, int(vae_patch_size)


def build_vae_from_config(config: Dict[str, Any]) -> Tuple[torch.nn.Module, Dict[str, Any], int]:
    cfg = strip_diffusers_metadata(copy.deepcopy(config))
    model = VQModel(**cfg).eval()
    vae_patch_size = 2 ** (len(cfg["ddconfig"]["ch_mult"]) - 1)
    return model, cfg, int(vae_patch_size)


def build_projector(model_root: Path, hidden_size: int) -> torch.nn.Module:
    ae_config = _read_json(model_root / "ae_config.json")
    z_channels = int(ae_config["ddconfig"]["z_channels"])
    model = MLPconnector(z_channels, hidden_size, "gelu_pytorch_tanh").eval()
    return model


def build_projector_from_config(config: Dict[str, Any]) -> torch.nn.Module:
    cfg = strip_diffusers_metadata(copy.deepcopy(config))
    model = MLPconnector(int(cfg["in_dim"]), int(cfg["out_dim"]), str(cfg["hidden_act"])).eval()
    return model


def build_text_model_and_tokenizer(
    model_root: Path,
    torch_dtype: torch.dtype,
    *,
    attention_mode: str = "auto",
    rms_norm_function: str = "default",
) -> Tuple[Any, torch.nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), trust_remote_code=True)
    config = AutoConfig.from_pretrained(str(model_root), trust_remote_code=True)
    attn_impl = _normalize_text_attention_mode(attention_mode)
    if attn_impl is not None:
        try:
            setattr(config, "_attn_implementation", attn_impl)
        except Exception:
            pass
    try:
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    except TypeError:
        # Fallback for transformer versions where from_config has fewer kwargs.
        model = AutoModelForCausalLM.from_config(config)
        if hasattr(model, "to"):
            model = model.to(dtype=torch_dtype)
    _configure_text_model_runtime(
        model,
        config,
        attention_mode=attention_mode,
        rms_norm_function=rms_norm_function,
    )
    return tokenizer, model.eval(), config


def build_text_model_from_config_dict(
    config_dict: Dict[str, Any],
    torch_dtype: torch.dtype,
    *,
    attention_mode: str = "auto",
    rms_norm_function: str = "default",
) -> Tuple[torch.nn.Module, Any]:
    cfg_dict = copy.deepcopy(config_dict)
    if Qwen3Config is not None:
        config = Qwen3Config(**cfg_dict)
    else:
        config = AutoConfig.for_model(cfg_dict.get("model_type", "qwen3"), **cfg_dict)
    attn_impl = _normalize_text_attention_mode(attention_mode)
    if attn_impl is not None:
        try:
            setattr(config, "_attn_implementation", attn_impl)
        except Exception:
            pass

    try:
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_config(config)
        if hasattr(model, "to"):
            model = model.to(dtype=torch_dtype)
    _configure_text_model_runtime(
        model,
        config,
        attention_mode=attention_mode,
        rms_norm_function=rms_norm_function,
    )
    return model.eval(), config


def build_tokenizer_from_source(tokenizer_source: str):
    return AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
