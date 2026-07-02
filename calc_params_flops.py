#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from importlib.machinery import ModuleSpec
import sys
import types
from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
from torch import nn
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DINO_ROOT = PROJECT_ROOT / "dinov3"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DINO_ROOT) not in sys.path:
    sys.path.insert(0, str(DINO_ROOT))




def _ensure_torchvision_stub() -> None:
    try:
        import torchvision  # noqa: F401
        return
    except Exception:
        pass

    class _IdentityTransform:
        def __init__(self, *args, **kwargs):
            self.transforms = kwargs.get("transforms", None)

        def __call__(self, x):
            if self.transforms is not None:
                for t in self.transforms:
                    x = t(x)
            return x

    class _Compose(_IdentityTransform):
        def __init__(self, transforms):
            self.transforms = transforms

    class _InterpolationMode:
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"

    torchvision_module = types.ModuleType("torchvision")
    transforms_module = types.ModuleType("torchvision.transforms")
    v2_module = types.ModuleType("torchvision.transforms.v2")

    v2_module.Compose = _Compose
    v2_module.RandomResizedCrop = _IdentityTransform
    v2_module.RandomHorizontalFlip = _IdentityTransform
    v2_module.Resize = _IdentityTransform
    v2_module.RandomApply = _IdentityTransform
    v2_module.ColorJitter = _IdentityTransform
    v2_module.RandomGrayscale = _IdentityTransform
    v2_module.RandomSolarize = _IdentityTransform
    v2_module.ToImage = _IdentityTransform
    v2_module.ToDtype = _IdentityTransform
    v2_module.GaussianBlur = _IdentityTransform
    v2_module.Normalize = _IdentityTransform
    v2_module.CenterCrop = _IdentityTransform
    v2_module.InterpolationMode = _InterpolationMode

    transforms_module.v2 = v2_module
    torchvision_module.transforms = transforms_module

    torchvision_module.__spec__ = ModuleSpec("torchvision", loader=None)
    transforms_module.__spec__ = ModuleSpec("torchvision.transforms", loader=None)
    v2_module.__spec__ = ModuleSpec("torchvision.transforms.v2", loader=None)

    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module
    sys.modules["torchvision.transforms.v2"] = v2_module


def _ensure_icecream_stub() -> None:
    try:
        import icecream  # noqa: F401
        return
    except Exception:
        pass

    icecream_module = types.ModuleType("icecream")

    def ic(*args, **kwargs):
        del kwargs
        if len(args) == 1:
            return args[0]
        return args

    icecream_module.ic = ic
    sys.modules["icecream"] = icecream_module


def _ensure_torchmetrics_stub() -> None:
    torchmetrics_module = types.ModuleType("torchmetrics")

    class Metric(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def update(self, *args, **kwargs):
            return None

        def compute(self):
            return torch.tensor(0.0)

        def reset(self):
            return None

    torchmetrics_module.Metric = Metric
    sys.modules["torchmetrics"] = torchmetrics_module


def _normalize_student_cfg(student_cfg: MutableMapping[str, Any]) -> None:
    norm_alias = {
        "nn.LayerNorm": "layernorm",
        "torch.nn.LayerNorm": "layernorm",
        "LayerNorm": "layernorm",
    }
    ffn_alias = {
        "nn.Linear": "mlp",
        "torch.nn.Linear": "mlp",
        "linear": "mlp",
    }

    norm_layer = student_cfg.get("norm_layer")
    if isinstance(norm_layer, str):
        student_cfg["norm_layer"] = norm_alias.get(norm_layer, norm_layer.lower())
    else:
        student_cfg.setdefault("norm_layer", "layernorm")

    ffn_layer = student_cfg.get("ffn_layer")
    if isinstance(ffn_layer, str):
        student_cfg["ffn_layer"] = ffn_alias.get(ffn_layer, ffn_layer.lower())
    else:
        student_cfg.setdefault("ffn_layer", "mlp")

    rope_norm = student_cfg.get("pos_embed_rope_normalize_coords")
    if isinstance(rope_norm, bool):
        student_cfg["pos_embed_rope_normalize_coords"] = "separate" if rope_norm else "min"
    elif rope_norm is None:
        student_cfg["pos_embed_rope_normalize_coords"] = "separate"

    student_cfg.setdefault("drop_path_rate", 0.0)
    student_cfg.setdefault("layerscale", 1e-5)

    if isinstance(student_cfg.get("drop_path_rate"), str):
        student_cfg["drop_path_rate"] = float(student_cfg["drop_path_rate"])
    if isinstance(student_cfg.get("layerscale"), str):
        student_cfg["layerscale"] = float(student_cfg["layerscale"])
    student_cfg.setdefault("qkv_bias", True)
    student_cfg.setdefault("ffn_bias", True)
    student_cfg.setdefault("proj_bias", True)
    student_cfg.setdefault("n_storage_tokens", 0)
    student_cfg.setdefault("mask_k_bias", False)
    student_cfg.setdefault("untie_cls_and_patch_norms", False)
    student_cfg.setdefault("untie_global_and_local_cls_norm", False)
    student_cfg.setdefault("fp8_enabled", False)
    student_cfg.setdefault("fp8_filter", "")
    student_cfg.setdefault("in_chans", 3)
    student_cfg.setdefault("pos_embed_rope_dtype", "bf16")
    student_cfg.setdefault("patch_size", 16)
    student_cfg.setdefault("arch", student_cfg.get("arch", "dino_vitl16"))


def _prepare_dinov3_cfg(cfg: Any, *, backbone_type: str | None = None) -> dict[str, Any]:
    if isinstance(cfg, Mapping):
        cfg_dict = deepcopy(dict(cfg))
    else:
        raise TypeError("cfg must be a mapping")

    cfg_dict.setdefault("student", {})
    student_cfg = cfg_dict["student"]
    if not isinstance(student_cfg, MutableMapping):
        raise ValueError("cfg['student'] must be dict")

    if backbone_type is not None:
        student_cfg["arch"] = backbone_type
    elif "arch" not in student_cfg:
        raise ValueError("missing student.arch")

    _normalize_student_cfg(student_cfg)

    cfg_dict.setdefault("crops", {})
    crops_cfg = cfg_dict["crops"]
    if not isinstance(crops_cfg, MutableMapping):
        raise ValueError("cfg['crops'] must be dict")
    crops_cfg.setdefault("global_crops_size", 518)
    return cfg_dict


def _dict_to_namespace(data: Any) -> Any:
    if isinstance(data, MutableMapping):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in data.items()})
    if isinstance(data, list):
        return [_dict_to_namespace(v) for v in data]
    return data


def _ensure_utils_stub() -> None:
    # MAS/model/backbone_utils.py does: from utils import dict_to_namespace, prepare_dinov3_cfg
    if "utils" in sys.modules:
        return

    utils_module = types.ModuleType("utils")
    utils_module.dict_to_namespace = _dict_to_namespace
    utils_module.prepare_dinov3_cfg = _prepare_dinov3_cfg
    sys.modules["utils"] = utils_module




def _ensure_dinov3_eval_utils_stub() -> None:
    module_name = "dinov3.eval.utils"
    if module_name in sys.modules:
        return

    eval_utils_module = types.ModuleType(module_name)

    class ModelWithIntermediateLayers(nn.Module):
        def __init__(self, feature_model, n, autocast_ctx, reshape=False, return_class_token=True):
            super().__init__()
            self.feature_model = feature_model
            self.n = n
            self.autocast_ctx = autocast_ctx
            self.reshape = reshape
            self.return_class_token = return_class_token

        def forward(self, images):
            with torch.inference_mode():
                with self.autocast_ctx():
                    return self.feature_model.get_intermediate_layers(
                        images,
                        n=self.n,
                        reshape=self.reshape,
                        return_class_token=self.return_class_token,
                    )

    eval_utils_module.ModelWithIntermediateLayers = ModelWithIntermediateLayers
    sys.modules[module_name] = eval_utils_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Calculate Params (M) and FLOPs (G) for MAS Mask2Former")
    parser.add_argument("--config", type=str, default=str(CURRENT_DIR / "config_m2f_segmentation.yaml"), help="Model config path")
    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size")
    parser.add_argument("--height", type=int, default=512, help="Input height")
    parser.add_argument("--width", type=int, default=512, help="Input width")
    parser.add_argument("--fps-num-images", type=int, default=100, help="Number of random images for FPS benchmark")
    parser.add_argument("--fps-warmup", type=int, default=10, help="Warmup iterations before FPS benchmark")
    parser.add_argument("--gpu-id", type=int, default=3, help="GPU index, e.g. 3 for cuda:3")
    parser.add_argument("--no-load-weights", dest="load_weights", action="store_false", help="Do not load pretrained/finetuned weights from config")
    parser.set_defaults(load_weights=True)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Failed to parse config as dict")
    return cfg


def build_model(cfg: dict, device: torch.device, load_weights: bool):
    _ensure_torchvision_stub()
    _ensure_icecream_stub()
    _ensure_torchmetrics_stub()
    _ensure_utils_stub()
    _ensure_dinov3_eval_utils_stub()

    from MAS.mask2former.mask2former import DinoMask2Former
    from MAS.mask2former.utils import build_mask2former_config, resolve_interaction_indices
    from MAS.model.backbone_utils import build_backbone
    from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter

    work_cfg = deepcopy(cfg)
    if not load_weights:
        work_cfg["pretrained_weights"] = None
        work_cfg["finetuned_weights"] = None

    mask2former_config = build_mask2former_config(work_cfg)
    feature_order = work_cfg.get("mask2former", {}).get("feature_order", ["1", "2", "3", "4"])

    try:
        backbone = build_backbone(work_cfg, device)
    except NotImplementedError as exc:
        if load_weights:
            raise
        fallback_cfg = deepcopy(cfg)
        backbone = build_backbone(fallback_cfg, device)
    indices = resolve_interaction_indices(backbone, work_cfg)
    adapter = DINOv3_Adapter(backbone, interaction_indexes=indices).to(device)
    model = DinoMask2Former(adapter, mask2former_config, feature_order).to(device)
    model.eval()
    return model


def _count_named_params_by_prefixes(
    model: nn.Module,
    prefixes: tuple[str, ...],
    *,
    trainable_only: bool | None = None,
) -> int:
    count = 0
    for name, parameter in model.named_parameters():
        in_group = any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
        if not in_group:
            continue
        if trainable_only is True and not parameter.requires_grad:
            continue
        if trainable_only is False and parameter.requires_grad:
            continue
        count += parameter.numel()
    return count


def summarize_encoder_decoder_params(model: nn.Module) -> dict[str, int]:
    encoder_prefixes = ("feature_extractor",)
    decoder_prefixes = (
        "pixel_decoder",
        "transformer_module",
        "class_predictor",
    )

    encoder_total = _count_named_params_by_prefixes(model, encoder_prefixes)
    encoder_trainable = _count_named_params_by_prefixes(model, encoder_prefixes, trainable_only=True)
    encoder_frozen = _count_named_params_by_prefixes(model, encoder_prefixes, trainable_only=False)

    decoder_total = _count_named_params_by_prefixes(model, decoder_prefixes)
    decoder_trainable = _count_named_params_by_prefixes(model, decoder_prefixes, trainable_only=True)
    decoder_frozen = _count_named_params_by_prefixes(model, decoder_prefixes, trainable_only=False)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen_params = total_params - trainable_params
    other_params = total_params - encoder_total - decoder_total

    return {
        "encoder_total": encoder_total,
        "encoder_trainable": encoder_trainable,
        "encoder_frozen": encoder_frozen,
        "decoder_total": decoder_total,
        "decoder_trainable": decoder_trainable,
        "decoder_frozen": decoder_frozen,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "other_params": other_params,
    }


@torch.no_grad()
def calc_complexity(model: torch.nn.Module, input_tensor: torch.Tensor):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    if not hasattr(torch, "profiler"):
        raise RuntimeError("Your PyTorch version does not provide torch.profiler.")

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=False,
        ) as prof:
            _ = model(pixel_values=input_tensor)
            torch.cuda.synchronize()
    except TypeError as exc:
        raise RuntimeError(
            "Current torch.profiler.profile does not support with_flops=True. "
            "Please upgrade PyTorch or use another FLOPs tool."
        ) from exc

    total_flops = float(sum(item.flops for item in prof.key_averages()))
    return total_params, trainable_params, total_flops


@torch.no_grad()
def benchmark_fps(
    model: torch.nn.Module,
    device: torch.device,
    *,
    height: int,
    width: int,
    num_images: int = 100,
    warmup_iters: int = 10,
) -> tuple[float, float]:
    if num_images <= 0:
        raise ValueError("fps-num-images must be > 0")
    if warmup_iters < 0:
        raise ValueError("fps-warmup must be >= 0")

    inputs = torch.randn(num_images, 3, height, width, device=device)
    warmup_input = torch.randn(1, 3, height, width, device=device)

    model.eval()
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(pixel_values=warmup_input)
        torch.cuda.synchronize(device)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        total_elapsed_ms = 0.0

        for idx in range(num_images):
            image = inputs[idx : idx + 1]
            starter.record()
            _ = model(pixel_values=image)
            ender.record()
            torch.cuda.synchronize(device)
            total_elapsed_ms += float(starter.elapsed_time(ender))

    avg_time_sec = (total_elapsed_ms / 1000.0) / float(num_images)
    fps = 1.0 / avg_time_sec if avg_time_sec > 0 else float("inf")
    return avg_time_sec, fps


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if args.gpu_id >= torch.cuda.device_count():
        raise ValueError(f"gpu-id={args.gpu_id} out of range, gpu_count={torch.cuda.device_count()}")

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    cfg = load_config(args.config)
    model = build_model(cfg=cfg, device=device, load_weights=args.load_weights)

    input_tensor = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    total_params, trainable_params, total_flops = calc_complexity(model=model, input_tensor=input_tensor)
    param_summary = summarize_encoder_decoder_params(model)
    avg_time_sec, fps = benchmark_fps(
        model=model,
        device=device,
        height=args.height,
        width=args.width,
        num_images=args.fps_num_images,
        warmup_iters=args.fps_warmup,
    )

    print("Model: MAS DinoMask2Former")
    print(f"Device: {device}")
    print(f"Input: [{args.batch_size}, 3, {args.height}, {args.width}]")
    print(f"Load Weights: {args.load_weights}")
    print("Encoder = feature_extractor (DINOv3_Adapter + backbone)")
    print("Decoder = pixel_decoder + transformer_module + class_predictor")
    print(f"Params (M): {total_params / 1e6:.4f}")
    print(f"Trainable Params (M): {trainable_params / 1e6:.4f}")
    print(f"Frozen Params (M): {(total_params - trainable_params) / 1e6:.4f}")
    print(f"Encoder Params (M): {param_summary['encoder_total'] / 1e6:.4f}")
    print(f"Encoder Trainable Params (M): {param_summary['encoder_trainable'] / 1e6:.4f}")
    print(f"Encoder Frozen Params (M): {param_summary['encoder_frozen'] / 1e6:.4f}")
    print(f"Decoder Params (M): {param_summary['decoder_total'] / 1e6:.4f}")
    print(f"Decoder Trainable Params (M): {param_summary['decoder_trainable'] / 1e6:.4f}")
    print(f"Decoder Frozen Params (M): {param_summary['decoder_frozen'] / 1e6:.4f}")
    if param_summary["other_params"] != 0:
        print(f"Other Params (M): {param_summary['other_params'] / 1e6:.4f}")
    print(f"FLOPs (G): {total_flops / 1e9:.4f}")
    print(f"FPS Benchmark Images: {args.fps_num_images} (batch_size=1)")
    print(f"FPS Benchmark Input: [1, 3, {args.height}, {args.width}]")
    print(f"Average Inference Time per Image (ms): {avg_time_sec * 1000.0:.3f}")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
