import os
import sys
import types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping, MutableMapping
from typing import Any, Union
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from dinov3.distributed import enable as enable_distributed, is_enabled as dist_is_enabled


CfgLike = Union[str, Mapping[str, Any]]


def prepare_dinov3_cfg(cfg: CfgLike, *, backbone_type: str | None = None) -> dict[str, Any]:
    """
    读取并规范化 DINOv3 所需的配置。

    参数:
        cfg: dict 或 YAML 文件路径，必须包含 student/crops 字段。
        backbone_type: 可选，用于覆盖 cfg['student']['arch']。

    返回:
        整理后的配置 dict。
    """
    cfg_dict = _load_cfg_as_dict(cfg)

    cfg_dict.setdefault("student", {})
    student_cfg = cfg_dict["student"]
    if not isinstance(student_cfg, MutableMapping):
        raise ValueError("`cfg['student']` 必须是一个字典")

    if backbone_type is not None:
        student_cfg["arch"] = backbone_type
    elif "arch" not in student_cfg:
        raise ValueError("配置缺少 `student.arch`，请在 cfg 或参数中指定 DINOv3 架构")

    _normalize_student_cfg(student_cfg)

    cfg_dict.setdefault("crops", {})
    crops_cfg = cfg_dict["crops"]
    if not isinstance(crops_cfg, MutableMapping):
        raise ValueError("`cfg['crops']` 必须是一个字典")
    crops_cfg.setdefault("global_crops_size", 518)

    return cfg_dict


def dict_to_namespace(data: Any) -> Any:
    """
    将嵌套 dict/list 转换为 SimpleNamespace。
    """
    if isinstance(data, MutableMapping):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in data.items()})
    if isinstance(data, list):
        return [dict_to_namespace(v) for v in data]
    return data

def load_config(cfg_path: str = "config_eval.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def _load_cfg_as_dict(cfg: CfgLike) -> dict[str, Any]:
    if isinstance(cfg, str):
        with open(cfg, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
    elif isinstance(cfg, Mapping):
        cfg_dict = deepcopy(dict(cfg))
    else:
        raise TypeError("`cfg` 需为 dict 或指向 YAML 配置文件的路径")

    if not isinstance(cfg_dict, MutableMapping):
        raise ValueError("配置加载失败：cfg 需为 key/value 结构")
    return cfg_dict


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

    base = student_cfg.get("pos_embed_rope_base")
    if isinstance(base, bool):
        base = float(base) if base else None
        student_cfg["pos_embed_rope_base"] = base

    for key in ("pos_embed_rope_min_period", "pos_embed_rope_max_period"):
        value = student_cfg.get(key)
        if isinstance(value, bool) and not value:
            student_cfg[key] = None

    if student_cfg.get("pos_embed_rope_base") is not None and (
        student_cfg.get("pos_embed_rope_min_period") is not None
        or student_cfg.get("pos_embed_rope_max_period") is not None
    ):
        student_cfg["pos_embed_rope_min_period"] = None
        student_cfg["pos_embed_rope_max_period"] = None

    for key in (
        "pos_embed_rope_shift_coords",
        "pos_embed_rope_jitter_coords",
        "pos_embed_rope_rescale_coords",
    ):
        value = student_cfg.get(key)
        if isinstance(value, bool):
            student_cfg[key] = None if value is False else value

    student_cfg.setdefault("drop_path_rate", 0.0)
    student_cfg.setdefault("layerscale", 1e-5)
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


def _ensure_torch_fsdp_compat() -> None:
    try:
        import torch.distributed.fsdp as torch_fsdp  # type: ignore
    except (ImportError, ModuleNotFoundError, AttributeError):
        return
    if not hasattr(torch_fsdp, "register_fsdp_forward_method"):

        def _register_fsdp_forward_method(*args, **kwargs):
            return None

        torch_fsdp.register_fsdp_forward_method = _register_fsdp_forward_method  # type: ignore[attr-defined]

    module_name = "torch.distributed.fsdp._fully_shard"
    submodule_name = f"{module_name}._fsdp_state"
    if submodule_name not in sys.modules:
        try:
            from torch.distributed._composable.fsdp import _fsdp_state as composable_fsdp_state  # type: ignore[attr-defined]
        except (ImportError, ModuleNotFoundError, AttributeError):
            return

        fully_shard_pkg = sys.modules.get(module_name)
        if fully_shard_pkg is None:
            fully_shard_pkg = types.ModuleType(module_name)
            fully_shard_pkg.__path__ = []  # type: ignore[attr-defined]
            sys.modules[module_name] = fully_shard_pkg
            setattr(torch_fsdp, "_fully_shard", fully_shard_pkg)

        fsdp_state_module = types.ModuleType(submodule_name)
        for attr in ("FSDPState", "FSDPStateContext"):
            if hasattr(composable_fsdp_state, attr):
                setattr(fsdp_state_module, attr, getattr(composable_fsdp_state, attr))

        sys.modules[submodule_name] = fsdp_state_module
        setattr(fully_shard_pkg, "_fsdp_state", fsdp_state_module)


def _setup_device_and_distributed() -> tuple[torch.device, int, int, int, bool]:
    cuda_available = torch.cuda.is_available()
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size_env > 1

    if distributed:
        if not dist_is_enabled():
            enable_distributed(set_cuda_current_device=True)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda", 0) if cuda_available else torch.device("cpu")
        rank = 0
        world_size = 1
        local_rank = 0
        if torch.distributed.is_available() and not torch.distributed.is_initialized():
            backend = "nccl" if device.type == "cuda" else "gloo"
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            torch.distributed.init_process_group(backend=backend, rank=0, world_size=1)
            dist.barrier()
        return device, rank, world_size, local_rank, distributed

    return device, rank, world_size, local_rank, distributed


def _distributed_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


def _build_teacher_temp_schedule(cfg: dict, total_iters: int) -> np.ndarray:
    """
    生成 warmup 阶段线性增加的温度。
    剩余步骤保持 final 温度。
    确保数组长度等于 total_iters。
    返回 numpy 数组。
    """
    base = float(cfg.get("base", 0.04))
    final = float(cfg.get("final", base))
    warmup = int(cfg.get("warmup_iters", 0))
    warmup = max(0, min(warmup, total_iters))

    if total_iters <= 0:
        return np.array([], dtype=np.float32)

    if warmup > 0:
        # 线性warmup
        warmup_vals = np.linspace(base, final, warmup, endpoint=False, dtype=np.float32)
        remaining = total_iters - warmup
        if remaining > 0:
            # 后续步骤采用final作为恒定温度
            rest = np.full((remaining,), final, dtype=np.float32)
            schedule = np.concatenate([warmup_vals, rest])
        else:
            schedule = warmup_vals
    else:
        schedule = np.full((total_iters,), final, dtype=np.float32)

    if schedule.size < total_iters:
        # 补齐长度，防止前面拼接导致长度不足 total_iters
        pad = np.full((total_iters - schedule.size,), final, dtype=np.float32)
        schedule = np.concatenate([schedule, pad])
    return schedule


def _ensure_dir(path: str | Path) -> None:
    if not path:
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def visualize_patch_similarity(
    model: nn.Module,
    img: torch.Tensor,
    *,
    patch_coords: tuple[int, int] | None = None,
    layer: int | None = None,
    cmap: str = "viridis",
    normalize_mode: str = "global",
    show: bool = True,
):
    """
    将 ViT patch 特征之间的余弦相似度可视化为热力图。

    参数:
        model: 具备 `get_intermediate_layers` 接口的模型，或包含该接口的子模块（例如 `DualImageDINOv3`）。
        img: 单张图像，shape 为 [3, H, W]，需预先按 ImageNet 均值 / 方差归一化。
        patch_coords: 指定要展示的 patch 坐标 (i, j)，默认取特征图中心位置。
        layer: 指定使用的层索引，默认取最后一层 (n=1)。
        cmap: Matplotlib 中的 colormap 名称。
        show: 是否直接使用 Matplotlib 展示热力图。

    返回:
        viz: 归一化到 [0, 1] 的 RGB 可视化张量，形状 [h, w, h, w, 3]。
        cos: 归一化到 [0, 1] 的余弦相似度张量，形状 [h, w, h, w]。
              若 normalize_mode="local"，代表按局部最小最大重新缩放。
    """
    if img.dim() != 3 or img.size(0) != 3:
        raise ValueError("输入张量需为 [3, H, W] 的单张图像。")

    # 推断模型所在设备，优先使用 model.device，其次读取参数设备
    device = getattr(model, "device", None)
    if device is None and hasattr(model, "backbone"):
        device = getattr(model.backbone, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    img = img.unsqueeze(0).to(device)

    # 适配 DualImageDINOv3 这类封装模型
    target_module = model
    if not hasattr(target_module, "get_intermediate_layers"):
        target_module = getattr(model, "backbone", None)
    if target_module is None or not hasattr(target_module, "get_intermediate_layers"):
        raise AttributeError("传入的模型缺少 get_intermediate_layers 方法。")

    layer_selector: int | list[int]
    if layer is None:
        layer_selector = 1
    else:
        layer_selector = [int(layer)]

    with torch.no_grad():
        feats = target_module.get_intermediate_layers(
            img,
            n=layer_selector,
            reshape=True,
        )[0]

    _, dim, h, w = feats.shape
    feats = feats.squeeze(0).movedim(-3, -1).flatten(0, 1)  # [hw, dim]
    feats = F.normalize(feats, p=2, dim=-1)  # L2 归一化便于计算余弦相似度

    cos = feats @ feats.transpose(0, 1)
    cos = cos.unflatten(1, (h, w)).unflatten(0, (h, w))
    cos = (cos + 1.0) / 2.0

    mode = normalize_mode.lower()
    if mode == "local":
        cos_min = cos.amin()
        cos_max = cos.amax()
        denom = torch.clamp(cos_max - cos_min, min=1e-6)
        cos = (cos - cos_min) / denom
    elif mode not in {"global", ""}:
        raise ValueError("normalize_mode 仅支持 'global' 或 'local'")

    viz = plt.get_cmap(cmap)(cos.detach().cpu().numpy())[..., :3]

    if patch_coords is None:
        patch_coords = (h // 2, w // 2)
    i, j = patch_coords
    if not (0 <= i < h and 0 <= j < w):
        raise ValueError(f"patch 坐标 {(i, j)} 超出特征网格范围 (h={h}, w={w})。")

    if show:
        plt.imshow(viz[i, j])
        plt.title(f"Similarity of patch {(i, j)} to all other patches")

    return viz, cos.cpu()
