from __future__ import annotations
# Created by aistudynow.com

import gc
import hashlib
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

import comfy.model_management
import comfy.model_patcher
import folder_paths
try:
    from comfy.utils import ProgressBar as ComfyProgressBar
except Exception:  # pragma: no cover - depends on host ComfyUI build
    ComfyProgressBar = None

try:
    from tqdm.auto import tqdm as tqdm_auto
except Exception:  # pragma: no cover - optional
    tqdm_auto = None

from . import local_model

try:
    import comfy.ops as comfy_ops
except Exception:  # pragma: no cover - depends on host ComfyUI build
    comfy_ops = None


LOGGER = logging.getLogger(__name__)
TOKENIZER_REPO_FALLBACK = "BiliSakura/BitDance-14B-64x-diffusers"
PLUGIN_DIR = Path(__file__).resolve().parent
BUNDLED_TOKENIZER_DIRS = (
    PLUGIN_DIR / "tokenizer",
    PLUGIN_DIR / "assets" / "tokenizer",
)
TEXT_EMBED_CACHE_DIR = PLUGIN_DIR / "text_embed_cache"
_TEXT_EMBED_MEMORY_CACHE: Dict[str, torch.Tensor] = {}
MODEL_FOLDER_TYPES = ("diffusion_models", "unet", "checkpoints", "diffusion_model")
TEXT_ENCODER_FOLDER_TYPES = ("text_encoders", "clip", "checkpoints", "text_encoder")
VAE_FOLDER_TYPES = ("vae", "checkpoints")

IMAGE_SIZE_LIST = [
    [2048, 512],
    [1920, 512],
    [1536, 640],
    [1280, 768],
    [1152, 896],
    [1024, 1024],
    [896, 1152],
    [768, 1280],
    [640, 1536],
    [512, 1920],
    [512, 2048],
    [1024, 256],
    [896, 256],
    [640, 384],
    [512, 512],
    [384, 640],
    [256, 896],
    [256, 1024],
]

def _bitdance_resolution_label(height: int, width: int) -> str:
    return f"{int(width)}x{int(height)}"


BITDANCE_RESOLUTION_CHOICES = tuple(_bitdance_resolution_label(h, w) for h, w in IMAGE_SIZE_LIST)


def _parse_bitdance_resolution_label(label: str) -> Tuple[int, int]:
    try:
        width_s, height_s = str(label).lower().split("x", 1)
        width = int(width_s.strip())
        height = int(height_s.strip())
    except Exception as e:
        raise ValueError(
            f"Invalid BitDance resolution '{label}'. Expected WIDTHxHEIGHT (example: 1024x1024)."
        ) from e
    return height, width


SUPPORTED_FP8_DTYPES = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu")
    if hasattr(torch, name)
)


@dataclass
class BitDanceModelRuntime:
    root: Path
    vision_head: torch.nn.Module
    projector: torch.nn.Module
    parallel_num: int
    ps: int
    hidden_size: int


@dataclass
class BitDanceTextRuntime:
    root: Path
    tokenizer: Any
    llm_model: torch.nn.Module
    hidden_size: int


@dataclass
class BitDanceVAERuntime:
    root: Path
    vae: torch.nn.Module
    vae_patch_size: int
    ae_config: Dict[str, Any]


@dataclass
class BitDanceLatentRuntime:
    tokens: torch.Tensor
    h: int
    w: int
    ps: int


@dataclass
class BitDanceTextEmbedsRuntime:
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: Optional[torch.Tensor]
    text_runtime: Optional[BitDanceTextRuntime] = None
    positive_prompt: str = ""
    negative_prompt: str = ""


@dataclass
class BitDanceResolutionRuntime:
    height: int
    width: int


class BitDanceFP8ScaledLinear(torch.nn.Module):
    """
    Runtime linear that can store fp8 weights + scale and dequantize on-the-fly.
    This preserves compressed text-encoder weights at rest in VRAM/RAM.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype or torch.float16),
            requires_grad=False,
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype or torch.float16),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)
        self.register_buffer("weight_scale", None, persistent=True)
        self._fp8_enabled = False

    def set_weight(self, weight: torch.Tensor, weight_scale: Optional[torch.Tensor] = None):
        with torch.no_grad():
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            if weight_scale is None:
                self.weight_scale = None
                self._fp8_enabled = False
            else:
                self.weight_scale = weight_scale
                self._fp8_enabled = True

    def set_bias(self, bias: torch.Tensor):
        with torch.no_grad():
            self.bias = torch.nn.Parameter(bias, requires_grad=False)

    def _dequant_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        w = self.weight
        s = self.weight_scale
        if s is None:
            return w.to(device=device, dtype=dtype)
        q = w.to(device=device).float()
        scale = s.to(device=device).float()
        while scale.ndim < q.ndim:
            scale = scale.unsqueeze(-1)
        w = q * scale
        return w.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fp8_enabled and self.weight_scale is not None:
            compute_dtype = x.dtype if x.is_floating_point() else torch.float16
            weight = self._dequant_weight(x.device, compute_dtype)
            bias = self.bias
            if bias is not None:
                bias = bias.to(device=x.device, dtype=weight.dtype)
            return torch.nn.functional.linear(x, weight, bias)

        bias = self.bias
        if bias is not None and (bias.device != x.device or (x.is_floating_point() and bias.dtype != x.dtype)):
            bias = bias.to(device=x.device, dtype=x.dtype if x.is_floating_point() else bias.dtype)
        weight = self.weight
        if weight.device != x.device or (x.is_floating_point() and weight.dtype != x.dtype):
            weight = weight.to(device=x.device, dtype=x.dtype if x.is_floating_point() else weight.dtype)
        return torch.nn.functional.linear(x, weight, bias)


class BitDanceComfyModelAdapter(torch.nn.Module):
    """
    Minimal MODEL-compatible wrapper so the loader can output a native MODEL type.
    BitDance decoding is autoregressive and should be sampled with BitDanceSampler.
    """

    def __init__(self, runtime: BitDanceModelRuntime):
        super().__init__()
        self.runtime = runtime
        self.diffusion_model = runtime.vision_head
        self.manual_cast_dtype = torch.bfloat16

    def apply_model(self, *args, **kwargs):
        raise RuntimeError(
            "BitDance is not a UNet denoiser. Use BitDanceSampler/BitDanceDecode nodes."
        )

    def memory_required(self, input_shape=None):
        return 0

    def extra_conds(self, **kwargs):
        return {}


class BitDanceClipAdapter:
    """
    Placeholder CLIP-compatible output object for modular wiring.
    """

    def __init__(self, runtime: BitDanceTextRuntime):
        self.runtime = runtime

    def encode_from_tokens(self, *args, **kwargs):
        raise RuntimeError(
            "BitDance uses Qwen3 autoregressive embeddings. Use BitDanceSampler directly."
        )


class BitDanceVAEAdapter:
    def __init__(self, runtime: BitDanceVAERuntime):
        self.runtime = runtime

    def decode(self, x):
        return self.runtime.vae.decode(x)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _get_checkpoint_path(ckpt_name: str) -> Path:
    return _get_full_path_from_folder_types(("checkpoints",), ckpt_name)


def _safe_get_filename_list(folder_type: str) -> Iterable[str]:
    try:
        return folder_paths.get_filename_list(folder_type)
    except Exception:
        return []


def _get_filename_list_multi(folder_types: Tuple[str, ...]) -> list[str]:
    seen = set()
    out = []
    for folder_type in folder_types:
        for name in _safe_get_filename_list(folder_type):
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _get_full_path_from_folder_types(folder_types: Tuple[str, ...], filename: str) -> Path:
    errors = []
    for folder_type in folder_types:
        try:
            if hasattr(folder_paths, "get_full_path_or_raise"):
                return Path(folder_paths.get_full_path_or_raise(folder_type, filename))
            path = folder_paths.get_full_path(folder_type, filename)
            if path is not None:
                return Path(path)
        except Exception as e:
            errors.append(f"{folder_type}: {e}")
            continue
    details = "; ".join(errors) if errors else "no matching file found"
    raise FileNotFoundError(f"Could not resolve {filename} in {folder_types}. {details}")


def _resolve_model_root_from_path(file_path: Path) -> Path:
    ckpt_path = file_path
    candidates = []
    if ckpt_path.is_dir():
        candidates.append(ckpt_path)
    candidates.append(ckpt_path.parent)
    if ckpt_path.parent.parent != ckpt_path.parent:
        candidates.append(ckpt_path.parent.parent)

    for root in candidates:
        if (root / "ae_config.json").exists() and (root / "vision_head_config.json").exists():
            return root
    raise ValueError(
        f"{file_path.name} does not appear to belong to a BitDance model bundle "
        f"(expected ae_config.json + vision_head_config.json in root)."
    )


def _resolve_model_root_from_checkpoint(ckpt_name: str) -> Path:
    return _resolve_model_root_from_path(_get_checkpoint_path(ckpt_name))


def _try_resolve_model_root_from_checkpoint(ckpt_name: str) -> Optional[Path]:
    try:
        return _resolve_model_root_from_checkpoint(ckpt_name)
    except Exception:
        return None


def _try_resolve_model_root_from_path(file_path: Path) -> Optional[Path]:
    try:
        return _resolve_model_root_from_path(file_path)
    except Exception:
        return None


def _find_first_existing(root: Path, filenames: Iterable[str]) -> Optional[Path]:
    for name in filenames:
        path = root / name
        if path.exists():
            return path
    return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_scale_key(key: str) -> bool:
    return (
        key.endswith(".weight.scale")
        or key.endswith(".weight_scale")
        or key.endswith(".scale_weight")
        or key.endswith(".weight_scales")
        or key.endswith(".input_scale")
    )


def _is_aux_quant_key(key: str) -> bool:
    return key.endswith(".comfy_quant")


def _scale_key_candidates(weight_key: str) -> Tuple[str, ...]:
    return (
        f"{weight_key}.scale",
        weight_key.replace(".weight", ".weight.scale"),
        weight_key.replace(".weight", ".weight_scale"),
        weight_key.replace(".weight", ".scale_weight"),
        weight_key.replace(".weight", ".weight_scales"),
        f"{weight_key}_scale",
    )


def _rename_diffusers_to_comfy_key(key: str) -> str:
    out = key
    if out.startswith("unet."):
        out = out[5:]

    # Lightweight fallback mapping for diffusers-style names.
    if out.startswith("conv_in."):
        out = out.replace("conv_in.", "input_blocks.0.0.", 1)
    if out.startswith("conv_norm_out."):
        out = out.replace("conv_norm_out.", "out.0.", 1)
    if out.startswith("conv_out."):
        out = out.replace("conv_out.", "out.2.", 1)
    if out.startswith("time_embedding.linear_1."):
        out = out.replace("time_embedding.linear_1.", "time_embed.0.", 1)
    if out.startswith("time_embedding.linear_2."):
        out = out.replace("time_embedding.linear_2.", "time_embed.2.", 1)

    out = out.replace("down_blocks.", "input_blocks.")
    out = out.replace("mid_block.", "middle_block.")
    out = out.replace("up_blocks.", "output_blocks.")

    if not out.startswith("diffusion_model."):
        out = f"diffusion_model.{out}"
    return out


def _detect_diffusers_format(state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(
        k.startswith("unet.down_blocks.")
        or k.startswith("down_blocks.")
        or k.startswith("unet.mid_block.")
        for k in state_dict.keys()
    )


def _dequantize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    external_scales: Optional[Mapping[str, torch.Tensor]] = None,
    target_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[str, torch.Tensor], bool]:
    scales: Dict[str, torch.Tensor] = {}
    out: Dict[str, torch.Tensor] = {}

    has_mixed_scale_keys = False
    use_diffusers_mapping = _detect_diffusers_format(state_dict)
    warned_cast_only_fp8 = False

    for key, value in state_dict.items():
        if _is_scale_key(key):
            scales[key] = value
            has_mixed_scale_keys = True
        elif _is_aux_quant_key(key):
            continue
    if external_scales is not None:
        for key, value in external_scales.items():
            if _is_scale_key(key):
                scales[key] = value
                has_mixed_scale_keys = True

    for key, value in state_dict.items():
        if _is_scale_key(key) or _is_aux_quant_key(key):
            continue

        out_key = _rename_diffusers_to_comfy_key(key) if use_diffusers_mapping else key
        tensor = value

        if tensor.is_floating_point() and tensor.dtype in SUPPORTED_FP8_DTYPES:
            scale_tensor = None
            scale_key_used = None
            for candidate in _scale_key_candidates(key):
                if candidate in scales:
                    scale_tensor = scales[candidate]
                    scale_key_used = candidate
                    break
            if scale_tensor is not None:
                tensor = tensor.float()
                scale_tensor = scale_tensor.float()
                while scale_tensor.ndim < tensor.ndim:
                    scale_tensor = scale_tensor.unsqueeze(-1)
                # Comfy-style weight_scale and this wrapper's converter store dequant scales where:
                #   q = x / scale   ->   x ~= q * scale
                # Some external formats may use input_scale semantics (divide on load).
                if scale_key_used is not None and scale_key_used.endswith(".input_scale"):
                    tensor = tensor / scale_tensor
                else:
                    tensor = tensor * scale_tensor
            else:
                if not warned_cast_only_fp8:
                    LOGGER.warning(
                        "FP8 weights found without scale tensors. Assuming raw cast-only FP8 checkpoint "
                        "(will load, but quality/stability may be degraded)."
                    )
                    warned_cast_only_fp8 = True
                tensor = tensor.float()

        if tensor.is_floating_point() and tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)

        out[out_key] = tensor

    return out, has_mixed_scale_keys


def _load_state_dict(module: torch.nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool):
    kwargs = {"strict": strict}
    params = inspect.signature(module.load_state_dict).parameters
    if "assign" in params:
        kwargs["assign"] = True
    return module.load_state_dict(state_dict, **kwargs)


def _apply_manual_cast_hint(module: torch.nn.Module):
    if comfy_ops is None or not hasattr(comfy_ops, "manual_cast"):
        return
    if not hasattr(module, "manual_cast_dtype"):
        module.manual_cast_dtype = torch.bfloat16
    for child in module.modules():
        if hasattr(child, "comfy_cast_weights"):
            child.comfy_cast_weights = True


def _empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_component_state_dict(
    module: torch.nn.Module,
    file_path: Path,
    target_dtype: torch.dtype,
    *,
    strict: bool,
) -> None:
    raw_sd = load_safetensors(str(file_path), device="cpu")
    sd, has_mixed_scales = _dequantize_state_dict(raw_sd, target_dtype=target_dtype)
    if has_mixed_scales:
        _apply_manual_cast_hint(module)
    _load_state_dict(module, sd, strict=strict)
    del raw_sd, sd
    gc.collect()
    _empty_cuda_cache()


def _load_component_state_dict_from_raw(
    module: torch.nn.Module,
    raw_sd: Mapping[str, torch.Tensor],
    target_dtype: torch.dtype,
    *,
    strict: bool,
) -> None:
    sd, has_mixed_scales = _dequantize_state_dict(raw_sd, target_dtype=target_dtype)
    if has_mixed_scales:
        _apply_manual_cast_hint(module)
    _load_state_dict(module, sd, strict=strict)
    del sd
    gc.collect()
    _empty_cuda_cache()


def _strip_first_matching_prefix(key: str, prefixes: Tuple[str, ...]) -> str:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _stream_load_safetensors_into_module(
    module: torch.nn.Module,
    file_path: Path,
    *,
    target_dtype: torch.dtype,
    strict: bool,
    key_prefixes: Tuple[str, ...] = (),
    device_override: Optional[torch.device] = None,
) -> None:
    """
    Streaming loader to avoid materializing a huge CPU state_dict in RAM.
    This is especially important for merged BitDance text encoder checkpoints.
    """
    if device_override is not None:
        try:
            sample_param = next(module.parameters())
            if sample_param.device.type == "meta":
                module = module.to_empty(device=device_override).eval()
            else:
                module = module.to(device_override).eval()
        except StopIteration:
            module = module.to(device_override).eval()

    state_ref = module.state_dict()
    expected_keys = set(state_ref.keys())
    loaded_keys = set()
    unexpected = []

    with safe_open(str(file_path), framework="pt", device="cpu") as sf:
        all_keys = list(sf.keys())
        key_set = set(all_keys)

        warned_cast_only_fp8 = False
        has_mixed_scales = any(_is_scale_key(k) for k in key_set)

        for raw_key in all_keys:
            if _is_scale_key(raw_key) or _is_aux_quant_key(raw_key):
                continue

            key = _strip_first_matching_prefix(raw_key, key_prefixes)
            out_key = key
            if out_key not in expected_keys:
                unexpected.append(raw_key)
                continue

            tensor = sf.get_tensor(raw_key)

            if tensor.is_floating_point() and tensor.dtype in SUPPORTED_FP8_DTYPES:
                scale_tensor = None
                scale_key_used = None
                for candidate in _scale_key_candidates(raw_key):
                    if candidate in key_set:
                        scale_key_used = candidate
                        scale_tensor = sf.get_tensor(candidate)
                        break
                if scale_tensor is not None:
                    dest = state_ref[out_key]
                    deq_device = dest.device if getattr(dest, "is_cuda", False) else tensor.device
                    tensor = tensor.to(deq_device).float()
                    scale_tensor = scale_tensor.to(deq_device).float()
                    while scale_tensor.ndim < tensor.ndim:
                        scale_tensor = scale_tensor.unsqueeze(-1)
                    if scale_key_used is not None and scale_key_used.endswith(".input_scale"):
                        tensor = tensor / scale_tensor
                    else:
                        tensor = tensor * scale_tensor
                else:
                    if not warned_cast_only_fp8:
                        LOGGER.warning(
                            "FP8 weights found without scale tensors in %s. "
                            "Assuming raw cast-only FP8 checkpoint (quality/stability may degrade).",
                            file_path.name,
                        )
                        warned_cast_only_fp8 = True
                    tensor = tensor.float()

            dest = state_ref[out_key]
            if tensor.is_floating_point():
                # respect destination dtype (model params may already be moved to GPU/device)
                tensor = tensor.to(dtype=dest.dtype)
            if tensor.device != dest.device:
                tensor = tensor.to(device=dest.device)

            with torch.no_grad():
                dest.copy_(tensor)
            loaded_keys.add(out_key)

            del tensor

        if has_mixed_scales:
            _apply_manual_cast_hint(module)

    if strict:
        missing = sorted(expected_keys - loaded_keys)
        if missing or unexpected:
            raise RuntimeError(
                f"Streaming load mismatch for {file_path.name}: "
                f"{len(missing)} missing keys, {len(unexpected)} unexpected keys."
            )

    gc.collect()
    _empty_cuda_cache()


def _try_import_init_empty_weights():
    try:
        from accelerate import init_empty_weights  # type: ignore
        return init_empty_weights
    except Exception:
        return None


def _module_device_dtype(module: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    try:
        p = next(module.parameters())
        return p.device, p.dtype
    except StopIteration:
        return torch.device("cpu"), torch.float32


def _set_child_module(root: torch.nn.Module, module_name: str, new_module: torch.nn.Module) -> None:
    if not module_name:
        raise ValueError("Cannot replace root module directly with _set_child_module.")
    parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def _replace_qwen_linears_with_fp8(model: torch.nn.Module, *, skip_lm_head: bool = True) -> int:
    replacements: list[Tuple[str, torch.nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if skip_lm_head and (name == "lm_head" or name.endswith(".lm_head")):
                continue
            replacements.append((name, mod))

    for name, old in replacements:
        device = old.weight.device
        dtype = old.weight.dtype
        if device.type == "meta":
            # Keep meta-init for low memory skeleton build.
            new_device = device
        else:
            new_device = device
        new_mod = BitDanceFP8ScaledLinear(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            device=new_device,
            dtype=dtype,
        )
        _set_child_module(model, name, new_mod)
    return len(replacements)


def _set_param_or_buffer_by_key(
    root: torch.nn.Module,
    modules_map: Dict[str, torch.nn.Module],
    key: str,
    tensor: torch.Tensor,
    *,
    fallback_device: torch.device,
    target_dtype: torch.dtype,
) -> bool:
    if "." not in key:
        mod_name, attr = "", key
    else:
        mod_name, attr = key.rsplit(".", 1)
    mod = modules_map.get(mod_name, None) if mod_name else root
    if mod is None:
        return False

    if isinstance(mod, BitDanceFP8ScaledLinear):
        if attr == "weight":
            # Dense fallback into FP8Linear (for kept BF16/FP16 weights).
            w = tensor
            if w.is_floating_point() and w.dtype != target_dtype:
                w = w.to(dtype=target_dtype)
            if w.device != fallback_device:
                w = w.to(fallback_device)
            mod.set_weight(w, None)
            return True
        if attr == "bias":
            b = tensor
            if b.is_floating_point() and b.dtype != target_dtype:
                b = b.to(dtype=target_dtype)
            if b.device != fallback_device:
                b = b.to(fallback_device)
            mod.set_bias(b)
            return True
        if attr == "weight_scale":
            s = tensor.to(fallback_device)
            mod.weight_scale = s
            mod._fp8_enabled = True
            return True
        return False

    existing = getattr(mod, attr, None)
    if isinstance(existing, torch.nn.Parameter):
        dev = existing.device if existing.device.type != "meta" else fallback_device
        dt = existing.dtype if existing.device.type != "meta" else (
            target_dtype if tensor.is_floating_point() else tensor.dtype
        )
        out = tensor.to(device=dev, dtype=dt) if tensor.is_floating_point() else tensor.to(device=dev)
        setattr(mod, attr, torch.nn.Parameter(out, requires_grad=False))
        return True
    if attr in getattr(mod, "_buffers", {}):
        existing_buf = mod._buffers.get(attr, None)
        if isinstance(existing_buf, torch.Tensor):
            dev = existing_buf.device if existing_buf.device.type != "meta" else fallback_device
            dt = existing_buf.dtype if existing_buf.device.type != "meta" else (
                target_dtype if tensor.is_floating_point() else tensor.dtype
            )
            out = tensor.to(device=dev, dtype=dt) if tensor.is_floating_point() else tensor.to(device=dev)
        else:
            out = tensor.to(device=fallback_device, dtype=target_dtype) if tensor.is_floating_point() else tensor.to(device=fallback_device)
        mod._buffers[attr] = out
        return True
    return False


def _stream_load_qwen_text_encoder_fp8(
    llm_model: torch.nn.Module,
    text_file: Path,
    *,
    target_dtype: torch.dtype,
    target_device: torch.device,
    allow_unquantized_dense: bool = True,
) -> Dict[str, int]:
    modules_map = dict(llm_model.named_modules())
    stats = {
        "loaded": 0,
        "fp8_linear": 0,
        "dense_linear": 0,
        "other": 0,
        "unexpected": 0,
    }

    with safe_open(str(text_file), framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        key_set = set(keys)
        for raw_key in keys:
            if _is_scale_key(raw_key) or _is_aux_quant_key(raw_key):
                continue
            key = _strip_first_matching_prefix(raw_key, ("text_encoder.", "model.text_encoder."))
            if not key:
                continue

            mod_name, attr = key.rsplit(".", 1) if "." in key else ("", key)
            mod = modules_map.get(mod_name, llm_model if not mod_name else None)
            if mod is None:
                stats["unexpected"] += 1
                continue

            tensor = sf.get_tensor(raw_key)

            if (
                isinstance(mod, BitDanceFP8ScaledLinear)
                and attr == "weight"
                and tensor.is_floating_point()
                and tensor.dtype in SUPPORTED_FP8_DTYPES
            ):
                scale_tensor = None
                for cand in _scale_key_candidates(raw_key):
                    if cand in key_set:
                        scale_tensor = sf.get_tensor(cand)
                        break
                if scale_tensor is None:
                    # Raw-cast-only fp8 fallback still supported, but slower/less accurate.
                    LOGGER.warning(
                        "BitDanceQwenFP8: weight %s has FP8 data but no scale tensor. "
                        "Falling back to dense dequantized load for this layer.",
                        key,
                    )
                    dense = tensor.float()
                    if allow_unquantized_dense:
                        ok = _set_param_or_buffer_by_key(
                            llm_model,
                            modules_map,
                            key,
                            dense,
                            fallback_device=target_device,
                            target_dtype=target_dtype,
                        )
                        if ok:
                            stats["dense_linear"] += 1
                            stats["loaded"] += 1
                        else:
                            stats["unexpected"] += 1
                    continue

                weight_fp8 = tensor.to(target_device)
                scale = scale_tensor.to(target_device)
                mod.set_weight(weight_fp8, scale)
                stats["fp8_linear"] += 1
                stats["loaded"] += 1
                continue

            ok = _set_param_or_buffer_by_key(
                llm_model,
                modules_map,
                key,
                tensor,
                fallback_device=target_device,
                target_dtype=target_dtype,
            )
            if ok:
                if isinstance(mod, BitDanceFP8ScaledLinear) and attr == "weight":
                    stats["dense_linear"] += 1
                else:
                    stats["other"] += 1
                stats["loaded"] += 1
            else:
                stats["unexpected"] += 1

    gc.collect()
    _empty_cuda_cache()
    return stats


def _build_text_runtime_from_single_file_fp8(
    text_file: Path,
    *,
    target_dtype: torch.dtype,
    attention_mode: str = "auto",
    rms_norm_function: str = "default",
    load_device_name: str = "main_device",
) -> BitDanceTextRuntime:
    tokenizer = _load_tokenizer_for_single_file(text_file)
    text_target_device = _named_device(load_device_name)
    cfg = local_model.DEFAULT_BITDANCE_64X_DIFFUSERS_TEXT_ENCODER_CONFIG

    init_empty_weights = _try_import_init_empty_weights()
    if init_empty_weights is not None:
        LOGGER.info("BitDanceQwenFP8: building Qwen text encoder with meta init (accelerate).")
        # Recreate local_model builder logic with meta-init to avoid full dense allocation.
        cfg_dict = dict(cfg)
        if getattr(local_model, "Qwen3Config", None) is not None:
            config = local_model.Qwen3Config(**cfg_dict)
        else:
            config = local_model.AutoConfig.for_model(cfg_dict.get("model_type", "qwen3"), **cfg_dict)
        attn_impl = local_model._normalize_text_attention_mode(attention_mode)  # type: ignore[attr-defined]
        if attn_impl is not None:
            try:
                setattr(config, "_attn_implementation", attn_impl)
            except Exception:
                pass
        with init_empty_weights():
            try:
                llm_model = local_model.AutoModelForCausalLM.from_config(  # type: ignore[attr-defined]
                    config,
                    trust_remote_code=True,
                    torch_dtype=target_dtype,
                )
            except TypeError:
                llm_model = local_model.AutoModelForCausalLM.from_config(config)  # type: ignore[attr-defined]
        local_model._configure_text_model_runtime(  # type: ignore[attr-defined]
            llm_model,
            config,
            attention_mode=attention_mode,
            rms_norm_function=rms_norm_function,
        )
        llm_config = config
    else:
        LOGGER.warning(
            "BitDanceQwenFP8: accelerate.init_empty_weights not available. "
            "Falling back to standard model init (higher RAM during model skeleton build)."
        )
        llm_model, llm_config = local_model.build_text_model_from_config_dict(
            cfg,
            target_dtype,
            attention_mode=attention_mode,
            rms_norm_function=rms_norm_function,
        )

    replaced = _replace_qwen_linears_with_fp8(llm_model, skip_lm_head=True)
    if hasattr(llm_model, "lm_head"):
        try:
            llm_model.lm_head = torch.nn.Identity()
        except Exception:
            pass
    LOGGER.info("BitDanceQwenFP8: replaced %d linear layers with FP8-capable modules.", replaced)

    stats = _stream_load_qwen_text_encoder_fp8(
        llm_model,
        text_file,
        target_dtype=target_dtype,
        target_device=text_target_device,
    )
    LOGGER.info(
        "BitDanceQwenFP8: streamed weights loaded (fp8_linear=%d, dense_linear=%d, other=%d, unexpected=%d).",
        stats["fp8_linear"],
        stats["dense_linear"],
        stats["other"],
        stats["unexpected"],
    )

    hidden_size = int(getattr(llm_config, "hidden_size", cfg["hidden_size"]))
    return BitDanceTextRuntime(
        root=text_file.parent,
        tokenizer=tokenizer,
        llm_model=llm_model.eval(),
        hidden_size=hidden_size,
    )


def _extract_prefixed_substate(
    raw_sd: Mapping[str, torch.Tensor], prefixes: Tuple[str, ...]
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        for prefix in prefixes:
            if k.startswith(prefix):
                out[k[len(prefix) :]] = v
                break
    return out


def _split_main_model_state_dict(raw_sd: Mapping[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    diffusion_head_sd = _extract_prefixed_substate(
        raw_sd,
        (
            "diffusion_head.",
            "model.diffusion_head.",
            "main_model.diffusion_head.",
            "vision_head.",
        ),
    )
    projector_sd = _extract_prefixed_substate(
        raw_sd,
        (
            "projector.",
            "model.projector.",
            "main_model.projector.",
        ),
    )

    if not diffusion_head_sd:
        # Bare combined file style: diffusion head keys start with net.*
        diffusion_head_sd = {k: v for k, v in raw_sd.items() if k.startswith("net.")}
    if not projector_sd:
        # Bare combined file style: projector keys are fc1/fc2
        projector_sd = {
            k: v
            for k, v in raw_sd.items()
            if k.startswith("fc1.") or k.startswith("fc2.")
        }

    return diffusion_head_sd, projector_sd


def _find_tokenizer_source_near(path: Path) -> Optional[str]:
    for c in BUNDLED_TOKENIZER_DIRS:
        if (c / "tokenizer.json").exists():
            return str(c)

    candidates = [
        path.parent / "tokenizer",
        path.parent.parent / "tokenizer",
        path.parent / "original_bitdance" / "tokenizer",
    ]
    for c in candidates:
        if (c / "tokenizer.json").exists():
            return str(c)

    # Flat tokenizer files in same folder
    flat_candidates = [
        path.parent / "tokenizer.json",
        path.parent / "vocab.json",
        path.parent / "merges.txt",
    ]
    if any(p.exists() for p in flat_candidates):
        return str(path.parent)

    return None


def _download_tokenizer_source() -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "Tokenizer files are missing and huggingface_hub is not available. "
            "Place a BitDance tokenizer folder locally (tokenizer.json + merges/vocab)."
        ) from e

    snapshot_dir = snapshot_download(
        repo_id=TOKENIZER_REPO_FALLBACK,
        allow_patterns=["tokenizer/*"],
        local_dir=str(PLUGIN_DIR),
    )
    tokenizer_dir = PLUGIN_DIR / "tokenizer"
    if tokenizer_dir.exists():
        return str(tokenizer_dir)

    # Fallback if the hub client returned a cached snapshot path instead of local_dir.
    cached_tokenizer_dir = Path(snapshot_dir) / "tokenizer"
    if cached_tokenizer_dir.exists():
        return str(cached_tokenizer_dir)
    raise FileNotFoundError(
        f"Downloaded snapshot missing tokenizer folder in both {tokenizer_dir} and {cached_tokenizer_dir}"
    )


def _load_tokenizer_for_single_file(text_encoder_file: Path):
    tokenizer_source = _find_tokenizer_source_near(text_encoder_file)
    if tokenizer_source is None:
        LOGGER.info(
            "BitDance tokenizer not found next to %s. Downloading tokenizer from %s.",
            text_encoder_file,
            TOKENIZER_REPO_FALLBACK,
        )
        tokenizer_source = _download_tokenizer_source()
    return local_model.build_tokenizer_from_source(tokenizer_source)


def _bitdance_prompt_templates(positive_prompt: str, negative_prompt: str) -> Tuple[str, str]:
    cond_prompt = f"<|im_start|>user\n{positive_prompt}<|im_end|>\n<|im_start|>assistant\n"
    if negative_prompt.strip():
        uncond_prompt = f"<|im_start|>user\n{negative_prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        uncond_prompt = "<|im_start|>assistant\n"
    return cond_prompt, uncond_prompt


def _text_embed_cache_key(kind: str, prompt: str, hidden_size: int) -> str:
    raw = f"bitdance64x|{kind}|h{hidden_size}|{prompt.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _text_embed_cache_path(cache_key: str) -> Path:
    return TEXT_EMBED_CACHE_DIR / f"{cache_key}.pt"


def _load_cached_text_embed(cache_key: str) -> Optional[torch.Tensor]:
    if cache_key in _TEXT_EMBED_MEMORY_CACHE:
        return _TEXT_EMBED_MEMORY_CACHE[cache_key]

    path = _text_embed_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        tensor = torch.load(str(path), map_location="cpu")
        if isinstance(tensor, torch.Tensor):
            _TEXT_EMBED_MEMORY_CACHE[cache_key] = tensor
            return tensor
    except Exception as e:
        LOGGER.warning("Failed to load BitDance text embed cache %s: %s", path, e)
    return None


def _save_cached_text_embed(cache_key: str, tensor: torch.Tensor) -> None:
    try:
        TEXT_EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cpu_tensor = tensor.detach().to("cpu")
        _TEXT_EMBED_MEMORY_CACHE[cache_key] = cpu_tensor
        path = _text_embed_cache_path(cache_key)
        if not path.exists():
            torch.save(cpu_tensor, str(path))
    except Exception as e:
        LOGGER.warning("Failed to save BitDance text embed cache: %s", e)


def _move_bitdance_model_runtime_to_offload(model_runtime: Optional[BitDanceModelRuntime]) -> None:
    if model_runtime is None:
        return
    try:
        offload_device = comfy.model_management.unet_offload_device()
        model_runtime.vision_head = model_runtime.vision_head.to(offload_device).eval()
        model_runtime.projector = model_runtime.projector.to(offload_device).eval()
        _empty_cuda_cache()
    except Exception as e:
        LOGGER.debug("Could not offload BitDance model runtime: %s", e)


def _apply_loader_optimization(
    model_runtime: BitDanceModelRuntime,
    text_runtime: BitDanceTextRuntime,
    vae_runtime: BitDanceVAERuntime,
    optimization: str,
) -> None:
    if optimization == "balanced":
        return

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    try:
        if optimization == "text_encoder_on_gpu":
            LOGGER.info("BitDanceLoader: preloading text encoder to %s", load_device)
            text_runtime.llm_model = text_runtime.llm_model.to(load_device).eval()
            return

        if optimization == "all_on_gpu":
            LOGGER.info("BitDanceLoader: preloading model/text_encoder/vae to %s", load_device)
            model_runtime.vision_head = model_runtime.vision_head.to(load_device).eval()
            model_runtime.projector = model_runtime.projector.to(load_device).eval()
            text_runtime.llm_model = text_runtime.llm_model.to(load_device).eval()
            vae_runtime.vae = vae_runtime.vae.to(load_device).eval()
            return

        if optimization == "offload_all":
            LOGGER.info("BitDanceLoader: moving model/text_encoder/vae to offload device %s", offload_device)
            model_runtime.vision_head = model_runtime.vision_head.to(offload_device).eval()
            model_runtime.projector = model_runtime.projector.to(offload_device).eval()
            text_runtime.llm_model = text_runtime.llm_model.to(offload_device).eval()
            vae_runtime.vae = vae_runtime.vae.to(offload_device).eval()
            _empty_cuda_cache()
            return
    except Exception as e:
        LOGGER.warning("BitDanceLoader optimization '%s' failed: %s", optimization, e)


def _named_device(name: str) -> torch.device:
    if name == "main_device":
        return comfy.model_management.get_torch_device()
    return comfy.model_management.unet_offload_device()


def _apply_loader_device_selection(
    model_runtime: BitDanceModelRuntime,
    text_runtime: BitDanceTextRuntime,
    vae_runtime: BitDanceVAERuntime,
    *,
    model_load_device: str,
    text_encoder_load_device: str,
    vae_load_device: str,
) -> None:
    try:
        md = _named_device(model_load_device)
        td = _named_device(text_encoder_load_device)
        vd = _named_device(vae_load_device)
        LOGGER.info(
            "BitDanceLoader: placing components -> model:%s text:%s vae:%s",
            md,
            td,
            vd,
        )
        model_runtime.vision_head = model_runtime.vision_head.to(md).eval()
        model_runtime.projector = model_runtime.projector.to(md).eval()
        text_runtime.llm_model = text_runtime.llm_model.to(td).eval()
        vae_runtime.vae = vae_runtime.vae.to(vd).eval()
        _empty_cuda_cache()
    except Exception as e:
        LOGGER.warning("BitDanceLoader explicit device placement failed: %s", e)


def _load_llm_shards(
    model: torch.nn.Module,
    model_root: Path,
    target_dtype: torch.dtype,
) -> None:
    llm_dir = model_root / "llm"
    model_shards = sorted(llm_dir.glob("model-*-of-*.safetensors"))
    if not model_shards:
        fallback = _find_first_existing(model_root, ("llm/model.safetensors", "model.safetensors"))
        if fallback is None:
            raise FileNotFoundError(f"No LLM shard files found under {llm_dir}")
        model_shards = [fallback]

    for model_shard in model_shards:
        raw_model_sd = load_safetensors(str(model_shard), device="cpu")

        scales_shard_name = model_shard.name.replace("model-", "scales-")
        scales_path = model_shard.with_name(scales_shard_name)
        raw_scale_sd = load_safetensors(str(scales_path), device="cpu") if scales_path.exists() else {}

        shard_sd, has_mixed_scales = _dequantize_state_dict(
            raw_model_sd, external_scales=raw_scale_sd, target_dtype=target_dtype
        )
        if has_mixed_scales:
            _apply_manual_cast_hint(model)
        _load_state_dict(model, shard_sd, strict=False)

        del raw_model_sd, raw_scale_sd, shard_sd
        gc.collect()
        _empty_cuda_cache()


def _get_base_llm(llm_model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(llm_model, "model"):
        return llm_model.model
    return llm_model


def _get_embed_tokens(base_llm: torch.nn.Module, llm_model: torch.nn.Module):
    if hasattr(base_llm, "embed_tokens"):
        return base_llm.embed_tokens
    return llm_model.get_input_embeddings()


def _encode_bitdance_text_prompts(
    text_encoder: BitDanceTextRuntime,
    positive_prompt: str,
    negative_prompt: str,
    *,
    device_mode: str = "gpu",
    use_disk_cache: bool = False,
    force_offload: bool = True,
    model_to_offload: Optional[BitDanceModelRuntime] = None,
) -> BitDanceTextEmbedsRuntime:
    hidden_size = int(text_encoder.hidden_size)
    pos_key = _text_embed_cache_key("pos", positive_prompt, hidden_size)
    neg_key = _text_embed_cache_key("neg", negative_prompt, hidden_size)

    cond_emb = _load_cached_text_embed(pos_key) if use_disk_cache else None
    uncond_emb = _load_cached_text_embed(neg_key) if use_disk_cache else None

    if cond_emb is not None and uncond_emb is not None:
        return BitDanceTextEmbedsRuntime(
            prompt_embeds=cond_emb,
            negative_prompt_embeds=uncond_emb,
            text_runtime=text_encoder,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
        )

    if device_mode == "gpu":
        device_to = comfy.model_management.get_torch_device()
        _move_bitdance_model_runtime_to_offload(model_to_offload)
    else:
        device_to = torch.device("cpu")

    text_encoder.llm_model = text_encoder.llm_model.to(device_to).eval()
    llm_model = text_encoder.llm_model
    tokenizer = text_encoder.tokenizer
    base_llm = _get_base_llm(llm_model)
    embed_tokens = _get_embed_tokens(base_llm, llm_model)

    cond_prompt, uncond_prompt = _bitdance_prompt_templates(positive_prompt, negative_prompt)

    with torch.no_grad():
        if cond_emb is None:
            cond_ids = torch.tensor(
                tokenizer.encode(cond_prompt, add_special_tokens=False),
                device=device_to,
                dtype=torch.long,
            )
            cond_emb = embed_tokens(cond_ids).detach().to("cpu")
            if use_disk_cache:
                _save_cached_text_embed(pos_key, cond_emb)

        if uncond_emb is None:
            uncond_ids = torch.tensor(
                tokenizer.encode(uncond_prompt, add_special_tokens=False),
                device=device_to,
                dtype=torch.long,
            )
            uncond_emb = embed_tokens(uncond_ids).detach().to("cpu")
            if use_disk_cache:
                _save_cached_text_embed(neg_key, uncond_emb)

    if force_offload and device_mode == "gpu":
        try:
            offload_device = comfy.model_management.unet_offload_device()
            text_encoder.llm_model = text_encoder.llm_model.to(offload_device).eval()
            _empty_cuda_cache()
        except Exception as e:
            LOGGER.debug("Could not offload BitDance text encoder after encoding: %s", e)

    return BitDanceTextEmbedsRuntime(
        prompt_embeds=cond_emb,
        negative_prompt_embeds=uncond_emb,
        text_runtime=text_encoder,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
    )


def _token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0:
        raise ValueError(f"Tokenizer is missing required special token: {token}")
    return int(token_id)


def _cache_seq_len(past_key_values) -> int:
    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
        return int(past_key_values[0][0].shape[2])
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if hasattr(past_key_values, "key_cache"):
        return int(past_key_values.key_cache[0].shape[2])
    raise TypeError("Unsupported cache type from text model.")


def _build_pos_embed_1d(hidden_size: int, vae_patch_size: int, device: torch.device) -> torch.Tensor:
    max_len = 4096 // vae_patch_size
    dim = hidden_size // 2
    if dim % 2 != 0:
        raise ValueError("BitDance hidden_size must be divisible by 4.")
    omega = torch.arange(dim // 2, dtype=torch.float32, device=device)
    omega /= dim / 2.0
    omega = 1.0 / (10000 ** omega)
    pos = torch.arange(max_len, dtype=torch.float32, device=device)
    out = torch.einsum("m,d->md", pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _get_2d_embed(
    pos_embed_1d: torch.Tensor,
    hidden_size: int,
    h: int,
    w: int,
    ps: int,
) -> torch.Tensor:
    emb_v = pos_embed_1d[:h]
    emb_h = pos_embed_1d[:w]
    grid_v = emb_v.view(h, 1, hidden_size // 2).repeat(1, w, 1)
    grid_h = emb_h.view(1, w, hidden_size // 2).repeat(h, 1, 1)
    pos_embed = torch.cat([grid_h, grid_v], dim=-1)
    return rearrange(pos_embed, "(hh p1) (ww p2) c -> (hh ww p1 p2) c", p1=ps, p2=ps)


def _build_model_runtime_from_single_file(main_file: Path, target_dtype: torch.dtype) -> BitDanceModelRuntime:
    raw_sd = load_safetensors(str(main_file), device="cpu")
    diffusion_head_raw, projector_raw = _split_main_model_state_dict(raw_sd)

    if not diffusion_head_raw:
        raise ValueError(
            f"{main_file.name} does not contain diffusion_head weights. "
            "Expected keys like 'net.*' or 'diffusion_head.*'."
        )
    if not projector_raw:
        raise ValueError(
            f"{main_file.name} does not contain projector weights. "
            "Expected keys like 'fc1.*'/'fc2.*' or 'projector.*'."
        )

    diffusion_head_cfg = local_model.DEFAULT_BITDANCE_64X_DIFFUSERS_DIFFUSION_HEAD_CONFIG
    projector_cfg = local_model.DEFAULT_BITDANCE_64X_DIFFUSERS_PROJECTOR_CONFIG
    vision_head, vision_head_cfg = local_model.build_vision_head_from_config(diffusion_head_cfg)
    projector = local_model.build_projector_from_config(projector_cfg)

    _load_component_state_dict_from_raw(vision_head, diffusion_head_raw, target_dtype, strict=True)
    _load_component_state_dict_from_raw(projector, projector_raw, target_dtype, strict=True)

    parallel_num = int(vision_head_cfg["parallel_num"])
    ps = int(parallel_num ** 0.5)
    return BitDanceModelRuntime(
        root=main_file.parent,
        vision_head=vision_head.eval(),
        projector=projector.eval(),
        parallel_num=parallel_num,
        ps=ps,
        hidden_size=int(projector_cfg["out_dim"]),
    )


def _build_text_runtime_from_single_file(
    text_file: Path,
    target_dtype: torch.dtype,
    *,
    attention_mode: str = "auto",
    rms_norm_function: str = "default",
    quantization: str = "disabled",
    load_device_name: str = "main_device",
) -> BitDanceTextRuntime:
    tokenizer = _load_tokenizer_for_single_file(text_file)

    cfg = local_model.DEFAULT_BITDANCE_64X_DIFFUSERS_TEXT_ENCODER_CONFIG
    text_target_device = _named_device(load_device_name)
    init_empty_weights = _try_import_init_empty_weights()

    if init_empty_weights is not None:
        LOGGER.info("BitDanceLoader: building Qwen disabled-quantization text encoder with meta init (accelerate) to save CPU RAM.")
        cfg_dict = dict(cfg)
        if getattr(local_model, "Qwen3Config", None) is not None:
            config = local_model.Qwen3Config(**cfg_dict)
        else:
            config = local_model.AutoConfig.for_model(cfg_dict.get("model_type", "qwen3"), **cfg_dict)
        
        attn_impl = local_model._normalize_text_attention_mode(attention_mode)
        if attn_impl is not None:
            try:
                setattr(config, "_attn_implementation", attn_impl)
            except Exception:
                pass
                
        with init_empty_weights():
            try:
                llm_model = local_model.AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=True,
                    torch_dtype=target_dtype,
                )
            except TypeError:
                llm_model = local_model.AutoModelForCausalLM.from_config(config)
        
        local_model._configure_text_model_runtime(
            llm_model,
            config,
            attention_mode=attention_mode,
            rms_norm_function=rms_norm_function,
        )
        llm_config = config
    else:
        LOGGER.warning(
            "BitDanceLoader: accelerate.init_empty_weights not available for disabled-quantization. "
            "Falling back to standard dense model init (will cause huge CPU RAM spike)."
        )
        llm_model, llm_config = local_model.build_text_model_from_config_dict(
            cfg,
            target_dtype,
            attention_mode=attention_mode,
            rms_norm_function=rms_norm_function,
        )
        llm_model = llm_model.to(text_target_device).eval()

    if quantization != "disabled":
        LOGGER.info(
            "BitDance text encoder quantization='%s' selected. "
            "Current BitDance Qwen loader still dequantizes FP8 weights to %s during load.",
            quantization,
            str(target_dtype).replace("torch.", ""),
        )
    _stream_load_safetensors_into_module(
        llm_model,
        text_file,
        target_dtype=target_dtype,
        strict=False,
        key_prefixes=("text_encoder.", "model.text_encoder."),
        device_override=text_target_device,
    )

    hidden_size = int(getattr(llm_config, "hidden_size", cfg["hidden_size"]))
    return BitDanceTextRuntime(
        root=text_file.parent,
        tokenizer=tokenizer,
        llm_model=llm_model.eval(),
        hidden_size=hidden_size,
    )


def _build_vae_runtime_from_single_file(vae_file: Path, target_dtype: torch.dtype) -> BitDanceVAERuntime:
    ae_cfg_file = _load_json(vae_file.parent / "config.json")
    if ae_cfg_file and "ddconfig" in ae_cfg_file:
        ae_cfg = ae_cfg_file
    else:
        ae_cfg = local_model.DEFAULT_BITDANCE_64X_DIFFUSERS_AUTOENCODER_CONFIG

    vae, ae_config, vae_patch_size = local_model.build_vae_from_config(ae_cfg)
    raw_sd = load_safetensors(str(vae_file), device="cpu")
    raw_sd = _extract_prefixed_substate(raw_sd, ("autoencoder.", "vae.", "model.autoencoder.")) or raw_sd
    _load_component_state_dict_from_raw(vae, raw_sd, target_dtype, strict=True)
    return BitDanceVAERuntime(
        root=vae_file.parent,
        vae=vae.eval(),
        vae_patch_size=int(vae_patch_size),
        ae_config=ae_config,
    )


class BitDanceModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (_get_filename_list_multi(MODEL_FOLDER_TYPES),),
                "dtype": (["auto", "bf16", "fp16", "fp32"],),
            }
        }

    RETURN_TYPES = ("MODEL", "BITDANCE_MODEL")
    RETURN_NAMES = ("model", "bitdance_model")
    FUNCTION = "load_model"
    CATEGORY = "loaders/bitdance"

    def load_model(self, ckpt_name: str, dtype: str, attention_mode: str = "auto"):
        target_dtype = _resolve_dtype(dtype)
        local_model.set_vision_attention_mode(attention_mode)
        selected_path = _get_full_path_from_folder_types(MODEL_FOLDER_TYPES, ckpt_name)
        model_root = _try_resolve_model_root_from_path(selected_path)
        if model_root is not None:
            hidden_size = local_model.load_hidden_size(model_root)
            vision_head, vision_head_config = local_model.build_vision_head(model_root)
            projector = local_model.build_projector(model_root, hidden_size)

            vision_head_file = _find_first_existing(
                model_root, ("vision_head_fp8_e4m3fn.safetensors", "vision_head.safetensors")
            )
            if vision_head_file is None:
                raise FileNotFoundError("Missing vision_head weights in BitDance model folder.")
            _load_component_state_dict(vision_head, vision_head_file, target_dtype, strict=True)

            projector_file = _find_first_existing(
                model_root, ("projector_fp8_e4m3fn.safetensors", "projector.safetensors")
            )
            if projector_file is None:
                raise FileNotFoundError("Missing projector weights in BitDance model folder.")
            _load_component_state_dict(projector, projector_file, target_dtype, strict=True)

            parallel_num = int(vision_head_config["parallel_num"])
            ps = int(parallel_num ** 0.5)
            runtime = BitDanceModelRuntime(
                root=model_root,
                vision_head=vision_head.eval(),
                projector=projector.eval(),
                parallel_num=parallel_num,
                ps=ps,
                hidden_size=hidden_size,
            )
        else:
            runtime = _build_model_runtime_from_single_file(selected_path, target_dtype)

        adapter = BitDanceComfyModelAdapter(runtime)
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        patcher = comfy.model_patcher.ModelPatcher(
            adapter,
            load_device=load_device,
            offload_device=offload_device,
        )

        return (patcher, runtime)


class BitDanceTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (_get_filename_list_multi(TEXT_ENCODER_FOLDER_TYPES),),
                "dtype": (["auto", "bf16", "fp16", "fp32"],),
            }
        }

    RETURN_TYPES = ("CLIP", "BITDANCE_TEXT_ENCODER")
    RETURN_NAMES = ("clip", "bitdance_text_encoder")
    FUNCTION = "load_text_encoder"
    CATEGORY = "loaders/bitdance"

    def load_text_encoder(
        self,
        ckpt_name: str,
        dtype: str,
        attention_mode: str = "auto",
        rms_norm_function: str = "default",
        quantization: str = "disabled",
        load_device_name: str = "main_device",
    ):
        target_dtype = _resolve_dtype(dtype)
        selected_path = _get_full_path_from_folder_types(TEXT_ENCODER_FOLDER_TYPES, ckpt_name)
        model_root = _try_resolve_model_root_from_path(selected_path)
        use_fp8_runtime = str(quantization or "disabled") in {
            "fp8_e4m3fn_scaled",
            "fp8_e4m3fn",
        }
        if model_root is not None:
            if use_fp8_runtime:
                LOGGER.warning(
                    "BitDanceQwenFP8 runtime is currently implemented for single-file merged text encoder checkpoints. "
                    "Bundle/sharded text encoder will use standard HF runtime."
                )
            tokenizer, llm_model, llm_config = local_model.build_text_model_and_tokenizer(
                model_root,
                target_dtype,
                attention_mode=attention_mode,
                rms_norm_function=rms_norm_function,
            )
            if quantization != "disabled":
                LOGGER.info(
                    "BitDance text encoder quantization='%s' selected. "
                    "Bundle loader still dequantizes FP8/scale weights to %s during load.",
                    quantization,
                    str(target_dtype).replace("torch.", ""),
                )
            _load_llm_shards(llm_model, model_root, target_dtype)
            # Place text encoder after loading (bundle mode still loads shards via CPU dict path).
            try:
                llm_model = llm_model.to(_named_device(load_device_name)).eval()
            except Exception as e:
                LOGGER.warning("Failed to place BitDance text encoder on %s: %s", load_device_name, e)

            hidden_size = int(getattr(llm_config, "hidden_size", local_model.load_hidden_size(model_root)))
            runtime = BitDanceTextRuntime(
                root=model_root,
                tokenizer=tokenizer,
                llm_model=llm_model.eval(),
                hidden_size=hidden_size,
            )
        else:
            if use_fp8_runtime:
                runtime = _build_text_runtime_from_single_file_fp8(
                    selected_path,
                    target_dtype=target_dtype,
                    attention_mode=attention_mode,
                    rms_norm_function=rms_norm_function,
                    load_device_name=load_device_name,
                )
            else:
                runtime = _build_text_runtime_from_single_file(
                    selected_path,
                    target_dtype,
                    attention_mode=attention_mode,
                    rms_norm_function=rms_norm_function,
                    quantization=quantization,
                    load_device_name=load_device_name,
                )
        return (BitDanceClipAdapter(runtime), runtime)


class BitDanceTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("BITDANCE_TEXT_ENCODER",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": False}),
                "model_to_offload": ("BITDANCE_MODEL",),
                "use_disk_cache": ("BOOLEAN", {"default": False}),
                "device": (["gpu", "cpu"], {"default": "gpu"}),
            },
        }

    RETURN_TYPES = ("BITDANCE_TEXT_EMBEDS", "BITDANCE_TEXT_EMBEDS", "STRING")
    RETURN_NAMES = ("positive", "negative", "positive_prompt")
    FUNCTION = "encode"
    CATEGORY = "conditioning/bitdance"

    def encode(
        self,
        text_encoder: BitDanceTextRuntime,
        positive_prompt: str,
        negative_prompt: str,
        force_offload: bool = True,
        model_to_offload: Optional[BitDanceModelRuntime] = None,
        use_disk_cache: bool = False,
        device: str = "gpu",
    ):
        embeds = _encode_bitdance_text_prompts(
            text_encoder,
            positive_prompt,
            negative_prompt,
            device_mode=device,
            use_disk_cache=use_disk_cache,
            force_offload=force_offload,
            model_to_offload=model_to_offload,
        )

        pos_only = BitDanceTextEmbedsRuntime(
            prompt_embeds=embeds.prompt_embeds,
            negative_prompt_embeds=None,
            text_runtime=text_encoder,
            positive_prompt=embeds.positive_prompt,
            negative_prompt=embeds.negative_prompt,
        )
        neg_only = BitDanceTextEmbedsRuntime(
            prompt_embeds=embeds.negative_prompt_embeds
            if embeds.negative_prompt_embeds is not None
            else embeds.prompt_embeds,
            negative_prompt_embeds=None,
            text_runtime=text_encoder,
            positive_prompt="",
            negative_prompt=embeds.negative_prompt,
        )
        return (pos_only, neg_only, embeds.positive_prompt)


class BitDanceTextEncodeCached:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("BITDANCE_TEXT_ENCODER",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": False}),
                "model_to_offload": ("BITDANCE_MODEL",),
                "device": (["gpu", "cpu"], {"default": "gpu"}),
            },
        }

    RETURN_TYPES = ("BITDANCE_TEXT_EMBEDS", "BITDANCE_TEXT_EMBEDS", "STRING")
    RETURN_NAMES = ("positive", "negative", "positive_prompt")
    FUNCTION = "encode_cached"
    CATEGORY = "conditioning/bitdance"

    def encode_cached(
        self,
        text_encoder: BitDanceTextRuntime,
        positive_prompt: str,
        negative_prompt: str,
        force_offload: bool = True,
        model_to_offload: Optional[BitDanceModelRuntime] = None,
        device: str = "gpu",
    ):
        embeds = _encode_bitdance_text_prompts(
            text_encoder,
            positive_prompt,
            negative_prompt,
            device_mode=device,
            use_disk_cache=True,
            force_offload=force_offload,
            model_to_offload=model_to_offload,
        )

        pos_only = BitDanceTextEmbedsRuntime(
            prompt_embeds=embeds.prompt_embeds,
            negative_prompt_embeds=None,
            text_runtime=text_encoder,
            positive_prompt=embeds.positive_prompt,
            negative_prompt=embeds.negative_prompt,
        )
        neg_only = BitDanceTextEmbedsRuntime(
            prompt_embeds=embeds.negative_prompt_embeds
            if embeds.negative_prompt_embeds is not None
            else embeds.prompt_embeds,
            negative_prompt_embeds=None,
            text_runtime=text_encoder,
            positive_prompt="",
            negative_prompt=embeds.negative_prompt,
        )
        return (pos_only, neg_only, embeds.positive_prompt)


class BitDanceVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (_get_filename_list_multi(VAE_FOLDER_TYPES),),
                "dtype": (["auto", "bf16", "fp16", "fp32"],),
            }
        }

    RETURN_TYPES = ("VAE", "BITDANCE_VAE")
    RETURN_NAMES = ("vae", "bitdance_vae")
    FUNCTION = "load_vae"
    CATEGORY = "loaders/bitdance"

    def load_vae(self, ckpt_name: str, dtype: str):
        target_dtype = _resolve_dtype(dtype)
        selected_path = _get_full_path_from_folder_types(VAE_FOLDER_TYPES, ckpt_name)
        model_root = _try_resolve_model_root_from_path(selected_path)
        if model_root is not None:
            vae, ae_config, vae_patch_size = local_model.build_vae(model_root)

            ae_file = _find_first_existing(model_root, ("ae_fp8_e4m3fn.safetensors", "ae.safetensors"))
            if ae_file is None:
                raise FileNotFoundError("Missing VAE weights in BitDance model folder.")
            _load_component_state_dict(vae, ae_file, target_dtype, strict=True)

            runtime = BitDanceVAERuntime(
                root=model_root,
                vae=vae.eval(),
                vae_patch_size=int(vae_patch_size),
                ae_config=ae_config,
            )
        else:
            runtime = _build_vae_runtime_from_single_file(selected_path, target_dtype)
        return (BitDanceVAEAdapter(runtime), runtime)


class BitDanceLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_get_filename_list_multi(MODEL_FOLDER_TYPES),),
                "text_encoder_name": (_get_filename_list_multi(TEXT_ENCODER_FOLDER_TYPES),),
                "vae_name": (_get_filename_list_multi(VAE_FOLDER_TYPES),),
                "quantization": (
                    ["disabled", "fp8_e4m3fn_scaled", "fp8_e4m3fn", "auto"],
                    {
                        "default": "fp8_e4m3fn_scaled",
                        "tooltip": (
                            "Loader quantization mode label. BitDance Qwen text encoder currently dequantizes "
                            "FP8 weights to target dtype during load (CPU+RAM first)."
                        ),
                    },
                ),
                "load_device": (
                    ["main_device", "offload_device"],
                    {"default": "offload_device", "tooltip": "Initial placement for BitDance main model (vision head + projector)."},
                ),
                "text_encoder_load_device": (
                    ["main_device", "offload_device"],
                    {"default": "main_device", "tooltip": "Placement for the BitDance text encoder after load completes."},
                ),
                "vae_load_device": (
                    ["main_device", "offload_device"],
                    {"default": "offload_device", "tooltip": "Initial placement for the BitDance VAE."},
                ),
                "attention_mode": (
                    ["auto", "sdpa", "flash_attn_2", "flash_attn_3", "eager"],
                    {"default": "auto", "tooltip": "Attention backend for BitDance vision head (and mapped to Qwen backend where supported)."},
                ),
                "rms_norm_function": (
                    ["default", "pytorch"],
                    {"default": "default", "tooltip": "Optional RMSNorm replacement for the Qwen text encoder. 'pytorch' may be faster but changes results slightly."},
                ),
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("BITDANCE_MODEL", "BITDANCE_TEXT_ENCODER", "BITDANCE_VAE")
    RETURN_NAMES = ("bitdance_model", "bitdance_text_encoder", "bitdance_vae")
    FUNCTION = "load"
    CATEGORY = "loaders/bitdance"

    def load(
        self,
        model_name: str,
        text_encoder_name: str,
        vae_name: str,
        quantization: str,
        load_device: str,
        text_encoder_load_device: str,
        vae_load_device: str,
        attention_mode: str,
        rms_norm_function: str,
        precision: str,
    ):
        model_loader = BitDanceModelLoader()
        text_loader = BitDanceTextEncoderLoader()
        vae_loader = BitDanceVAELoader()

        LOGGER.info(
            "BitDanceLoader: loading components (main=%s, text=%s, vae=%s). "
            "Text encoder single-file path uses streaming load to reduce CPU RAM spikes.",
            model_name,
            text_encoder_name,
            vae_name,
        )
        if quantization == "auto":
            quantization = "disabled"
        _, bitdance_model = model_loader.load_model(model_name, precision, attention_mode=attention_mode)
        _, bitdance_text_encoder = text_loader.load_text_encoder(
            text_encoder_name,
            precision,
            attention_mode=attention_mode,
            rms_norm_function=rms_norm_function,
            quantization=quantization,
            load_device_name=text_encoder_load_device,
        )
        _, bitdance_vae = vae_loader.load_vae(vae_name, precision)
        _apply_loader_device_selection(
            bitdance_model,
            bitdance_text_encoder,
            bitdance_vae,
            model_load_device=load_device,
            text_encoder_load_device=text_encoder_load_device,
            vae_load_device=vae_load_device,
        )

        return (bitdance_model, bitdance_text_encoder, bitdance_vae)


def _coerce_bitdance_resolution(resolution: Any) -> BitDanceResolutionRuntime:
    if isinstance(resolution, BitDanceResolutionRuntime):
        return BitDanceResolutionRuntime(height=int(resolution.height), width=int(resolution.width))
    if isinstance(resolution, Mapping):
        if "height" in resolution and "width" in resolution:
            return BitDanceResolutionRuntime(height=int(resolution["height"]), width=int(resolution["width"]))
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        return BitDanceResolutionRuntime(height=int(resolution[0]), width=int(resolution[1]))
    raise TypeError(
        "BitDanceSampler expected BITDANCE_RESOLUTION from BitDance Resolution node. "
        f"Got: {type(resolution).__name__}"
    )


class BitDanceResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (
                    list(BITDANCE_RESOLUTION_CHOICES),
                    {"default": _bitdance_resolution_label(1024, 1024)},
                ),
            }
        }

    RETURN_TYPES = ("BITDANCE_RESOLUTION", "INT", "INT", "STRING")
    RETURN_NAMES = ("resolution", "width", "height", "label")
    FUNCTION = "select"
    CATEGORY = "utils/bitdance"

    def select(self, resolution: str):
        height, width = _parse_bitdance_resolution_label(resolution)
        if [height, width] not in IMAGE_SIZE_LIST:
            raise ValueError(f"Unsupported BitDance resolution: {resolution}")
        out = BitDanceResolutionRuntime(height=height, width=width)
        return (out, int(width), int(height), _bitdance_resolution_label(height, width))


class BitDanceSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BITDANCE_MODEL",),
                "vae": ("BITDANCE_VAE",),
                "positive": ("BITDANCE_TEXT_EMBEDS",),
                "negative": ("BITDANCE_TEXT_EMBEDS",),
                "resolution": ("BITDANCE_RESOLUTION",),
                "sampler_name": (["euler_maruyama", "euler"], {"default": "euler_maruyama"}),
                "num_sampling_steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("BITDANCE_LATENT",)
    RETURN_NAMES = ("bitdance_latent",)
    FUNCTION = "sample"
    CATEGORY = "sampling/bitdance"

    def sample(
        self,
        model: BitDanceModelRuntime,
        vae: BitDanceVAERuntime,
        positive: BitDanceTextEmbedsRuntime,
        negative: BitDanceTextEmbedsRuntime,
        resolution: BitDanceResolutionRuntime,
        sampler_name: str,
        num_sampling_steps: int,
        guidance_scale: float,
        num_images: int,
        seed: int,
    ):
        resolution = _coerce_bitdance_resolution(resolution)
        height = int(resolution.height)
        width = int(resolution.width)
        if [height, width] not in IMAGE_SIZE_LIST:
            raise ValueError(f"Unsupported image size {[height, width]}; choose one from BitDance IMAGE_SIZE_LIST.")

        vae_patch_size = int(vae.vae_patch_size)
        if height % vae_patch_size != 0 or width % vae_patch_size != 0:
            raise ValueError(f"height/width must be divisible by vae_patch_size={vae_patch_size}")

        max_length = (height // vae_patch_size) * (width // vae_patch_size)
        step_width = int(model.parallel_num)
        if max_length % step_width != 0:
            raise ValueError(
                f"Token count ({max_length}) must be divisible by parallel_num ({step_width})."
            )

        device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        text_encoder = positive.text_runtime or negative.text_runtime
        if text_encoder is None:
            raise ValueError(
                "BitDanceSampler requires positive/negative embeds produced by "
                "BitDanceTextEncode or BitDanceTextEncodeCached."
            )

        model.vision_head = model.vision_head.to(device).eval()
        model.projector = model.projector.to(device).eval()
        LOGGER.info("BitDanceSampler: moving text encoder runtime to %s (this can take time for 14B).", device)
        text_encoder.llm_model = text_encoder.llm_model.to(device).eval()

        tokenizer = text_encoder.tokenizer
        llm_model = text_encoder.llm_model
        base_llm = _get_base_llm(llm_model)
        embed_tokens = _get_embed_tokens(base_llm, llm_model)
        embed_weight = getattr(embed_tokens, "weight", None)
        embed_dtype = embed_weight.dtype if isinstance(embed_weight, torch.Tensor) else None

        h = height // vae_patch_size
        w = width // vae_patch_size
        ps = int(model.ps)
        hidden_size = int(text_encoder.hidden_size)

        autocast_enabled = device.type == "cuda"
        autocast_dtype = torch.bfloat16 if autocast_enabled else torch.float32

        LOGGER.info(
            "BitDanceSampler: start sampling %dx%d, blocks=%d, diffusion_steps=%d, cfg=%.2f, images=%d",
            width,
            height,
            max_length // step_width,
            num_sampling_steps,
            guidance_scale,
            num_images,
        )
        LOGGER.info("BitDanceSampler: sampler=%s", sampler_name)
        overall_inner_steps = (max_length // step_width) * int(num_sampling_steps)
        comfy_progress = ComfyProgressBar(int(num_sampling_steps)) if (ComfyProgressBar is not None and int(num_sampling_steps) > 0) else None
        overall_inner_done = 0
        shared_tqdm_bar = None
        if tqdm_auto is not None and int(num_sampling_steps) > 0:
            try:
                shared_tqdm_bar = tqdm_auto(
                    total=int(num_sampling_steps),
                    desc="BitDanceSampler",
                    leave=True,
                    dynamic_ncols=True,
                )
            except Exception:
                shared_tqdm_bar = None
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=autocast_enabled, dtype=autocast_dtype
        ):
            cond_emb = positive.prompt_embeds.to(device=device)
            if embed_dtype is not None and cond_emb.dtype != embed_dtype:
                cond_emb = cond_emb.to(dtype=embed_dtype)
            if guidance_scale > 1.0:
                uncond_emb = negative.prompt_embeds.to(device=device)
                if embed_dtype is not None and uncond_emb.dtype != embed_dtype:
                    uncond_emb = uncond_emb.to(dtype=embed_dtype)

            img_start_id = _token_id(tokenizer, "<|vision_start|>")
            res_h_id = _token_id(tokenizer, f"<|res_{h}|>")
            res_w_id = _token_id(tokenizer, f"<|res_{w}|>")
            img_start_emb = embed_tokens(
                torch.tensor([img_start_id, res_h_id, res_w_id], device=device, dtype=torch.long)
            )

            for i in range(1, model.parallel_num):
                query_id = _token_id(tokenizer, f"<|query_{i}|>")
                query_embed = embed_tokens(torch.tensor([query_id], device=device, dtype=torch.long))
                img_start_emb = torch.cat([img_start_emb, query_embed], dim=0)

            pos_embed_1d = _build_pos_embed_1d(hidden_size, vae_patch_size, device)
            pos_embed_for_diff = _get_2d_embed(pos_embed_1d, hidden_size, h, w, ps).unsqueeze(0)

            input_embeds_cond = torch.cat([cond_emb, img_start_emb], dim=0).unsqueeze(0).repeat(num_images, 1, 1).to(dtype=autocast_dtype)
            outputs_c = base_llm(inputs_embeds=input_embeds_cond[:, :-step_width, :], use_cache=True)
            pkv_c = outputs_c.past_key_values

            pkv_len = _cache_seq_len(pkv_c)
            bi_attn_mask = torch.ones(
                (num_images, 1, step_width, step_width + pkv_len),
                dtype=torch.bool,
                device=device,
            )
            outputs_c = base_llm(
                inputs_embeds=input_embeds_cond[:, -step_width:, :],
                past_key_values=pkv_c,
                use_cache=True,
                attention_mask=bi_attn_mask,
            )
            pkv_c = outputs_c.past_key_values
            hidden_c = outputs_c.last_hidden_state[:, -step_width:]

            if guidance_scale > 1.0:
                input_embeds_uncond = torch.cat([uncond_emb, img_start_emb], dim=0).unsqueeze(0).repeat(num_images, 1, 1).to(dtype=autocast_dtype)
                outputs_u = base_llm(inputs_embeds=input_embeds_uncond[:, :-step_width, :], use_cache=True)
                pkv_u = outputs_u.past_key_values
                outputs_u = base_llm(
                    inputs_embeds=input_embeds_uncond[:, -step_width:, :],
                    past_key_values=pkv_u,
                    use_cache=True,
                    attention_mask=bi_attn_mask,
                )
                pkv_u = outputs_u.past_key_values
                hidden_u = outputs_u.last_hidden_state[:, -step_width:]

            out_tokens = []
            num_steps = max_length // step_width
            for step in range(num_steps):
                h_fused = torch.cat([hidden_c, hidden_u], dim=0) if guidance_scale > 1.0 else hidden_c
                h_fused = h_fused + pos_embed_for_diff[:, step * step_width : (step + 1) * step_width, :]

                tqdm_bar = shared_tqdm_bar
                inner_last_step = 0
                
                if tqdm_bar is not None:
                    # Do not reset for every chunk, just update continuously based on overall progress
                    pass

                def _inner_step_progress(cur_step: int, total_steps: int):
                    nonlocal inner_last_step, overall_inner_done
                    delta = max(0, int(cur_step) - int(inner_last_step))
                    if delta == 0:
                        return
                    inner_last_step = int(cur_step)
                    if tqdm_bar is not None:
                        try:
                            # Update tqdm based on fractional block completion
                            tqdm_bar.update(delta / num_steps)
                        except Exception:
                            pass
                    if comfy_progress is not None:
                        overall_inner_done += delta
                        try:
                            comfy_progress.update_absolute(int(overall_inner_done / num_steps), int(num_sampling_steps))
                        except Exception:
                            try:
                                comfy_progress.update(int(delta / num_steps))
                            except Exception:
                                pass

                pred_latents = model.vision_head.sample(
                    h_fused,
                    cfg=guidance_scale,
                    num_sampling_steps=num_sampling_steps,
                    sampler_name=sampler_name,
                    progress_callback=_inner_step_progress,
                )
                if tqdm_bar is not None:
                    try:
                        # Ensure any remaining steps for this block are pushed
                        if inner_last_step < int(num_sampling_steps):
                            tqdm_bar.update((int(num_sampling_steps) - inner_last_step) / num_steps)
                    except Exception:
                        pass
                curr_tokens = torch.sign(pred_latents)
                curr_embeds = model.projector(curr_tokens)
                out_tokens.append(curr_tokens[:num_images])

                model_input = curr_embeds + pos_embed_for_diff[:, step * step_width : (step + 1) * step_width, :]
                model_input = model_input.to(dtype=autocast_dtype)
                pkv_len = _cache_seq_len(pkv_c)
                bi_attn_mask = torch.ones(
                    (model_input.shape[0], 1, model_input.shape[1], model_input.shape[1] + pkv_len),
                    dtype=torch.bool,
                    device=device,
                )

                outputs_c = base_llm(
                    inputs_embeds=model_input[:num_images],
                    past_key_values=pkv_c,
                    use_cache=True,
                    attention_mask=bi_attn_mask[:num_images],
                )
                pkv_c = outputs_c.past_key_values
                hidden_c = outputs_c.last_hidden_state[:, -step_width:]

                if guidance_scale > 1.0:
                    outputs_u = base_llm(
                        inputs_embeds=model_input[num_images:],
                        past_key_values=pkv_u,
                        use_cache=True,
                        attention_mask=bi_attn_mask[num_images:],
                    )
                    pkv_u = outputs_u.past_key_values
                    hidden_u = outputs_u.last_hidden_state[:, -step_width:]

            full_output = torch.cat(out_tokens, dim=1)

        if shared_tqdm_bar is not None:
            try:
                shared_tqdm_bar.close()
            except Exception:
                pass

        model.vision_head = model.vision_head.to(offload_device)
        model.projector = model.projector.to(offload_device)
        text_encoder.llm_model = text_encoder.llm_model.to(offload_device)
        comfy.model_management.soft_empty_cache()

        latent = BitDanceLatentRuntime(tokens=full_output[:num_images].detach(), h=h, w=w, ps=ps)
        return (latent,)


class BitDanceDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("BITDANCE_VAE",),
                "bitdance_latent": ("BITDANCE_LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "latent/bitdance"

    def decode(self, vae: BitDanceVAERuntime, bitdance_latent: BitDanceLatentRuntime):
        device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        vae.vae = vae.vae.to(device).eval()

        tokens = bitdance_latent.tokens.to(device)
        h = int(bitdance_latent.h)
        w = int(bitdance_latent.w)
        ps = int(bitdance_latent.ps)

        if h % ps != 0 or w % ps != 0:
            raise ValueError(f"Invalid latent grid: h={h}, w={w}, ps={ps}")

        image_latents = rearrange(
            tokens,
            "b (hh ww p1 p2) c -> b c (hh p1) (ww p2)",
            hh=h // ps,
            ww=w // ps,
            p1=ps,
            p2=ps,
        )

        autocast_enabled = device.type == "cuda"
        autocast_dtype = torch.bfloat16 if autocast_enabled else torch.float32
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=autocast_dtype):
            decoded = vae.vae.decode(image_latents)

        image = torch.clamp(decoded, -1.0, 1.0)
        image = (image + 1.0) * 0.5
        image = image.permute(0, 2, 3, 1).to(dtype=torch.float32).cpu()

        vae.vae = vae.vae.to(offload_device)
        comfy.model_management.soft_empty_cache()

        return (image,)


class BitDanceEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("BITDANCE_VAE",),
                "image": ("IMAGE",),
                "parallel_num": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("BITDANCE_LATENT",)
    RETURN_NAMES = ("bitdance_latent",)
    FUNCTION = "encode"
    CATEGORY = "latent/bitdance"

    def encode(self, vae: BitDanceVAERuntime, image: torch.Tensor, parallel_num: int):
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("BitDanceEncode expects IMAGE tensor with shape [B, H, W, 3].")

        ps = int(round(float(parallel_num) ** 0.5))
        if ps * ps != int(parallel_num):
            raise ValueError(f"parallel_num must be a perfect square, got {parallel_num}.")

        device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        vae.vae = vae.vae.to(device).eval()

        x = image.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        x = (x * 2.0) - 1.0

        autocast_enabled = device.type == "cuda"
        autocast_dtype = torch.bfloat16 if autocast_enabled else torch.float32
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=autocast_dtype):
            quant = vae.vae.encode(x)

        if isinstance(quant, (tuple, list)):
            quant = quant[0]
        if not isinstance(quant, torch.Tensor) or quant.ndim != 4:
            raise TypeError("Unexpected BitDance VAE encode output. Expected BCHW tensor.")

        h = int(quant.shape[-2])
        w = int(quant.shape[-1])
        if h % ps != 0 or w % ps != 0:
            raise ValueError(
                f"Encoded latent grid ({h}x{w}) is not divisible by ps={ps}. "
                "Use a supported image resolution for BitDance."
            )

        tokens = rearrange(
            quant,
            "b c (hh p1) (ww p2) -> b (hh ww p1 p2) c",
            p1=ps,
            p2=ps,
        )

        vae.vae = vae.vae.to(offload_device)
        comfy.model_management.soft_empty_cache()

        return (BitDanceLatentRuntime(tokens=tokens.detach(), h=h, w=w, ps=ps),)


NODE_CLASS_MAPPINGS = {
    "BitDanceLoader": BitDanceLoader,
    "BitDanceResolution": BitDanceResolution,
    "BitDanceTextEncode": BitDanceTextEncode,
    "BitDanceTextEncodeCached": BitDanceTextEncodeCached,
    "BitDanceSampler": BitDanceSampler,
    "BitDanceDecode": BitDanceDecode,
    "BitDanceEncode": BitDanceEncode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "BitDanceLoader": "BitDance Loader",
    "BitDanceResolution": "BitDance Resolution",
    "BitDanceTextEncode": "BitDance Text Encode",
    "BitDanceTextEncodeCached": "BitDance Text Encode Cached",
    "BitDanceSampler": "BitDance Sampler",
    "BitDanceDecode": "BitDance VAE Decode",
    "BitDanceEncode": "BitDance VAE Encode",
}
