import os
import sys
import types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping, MutableMapping
from typing import Any, Union, Optional
import numpy as np
import torch
import torch.distributed as dist
import yaml

_CURRENT_DIR = Path(__file__).resolve().parent
_DINOV3_ROOT = _CURRENT_DIR.parent / "dinov3"
if _DINOV3_ROOT.exists() and str(_DINOV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_DINOV3_ROOT))

from dinov3.distributed import (
    enable as enable_distributed,
    is_enabled as dist_is_enabled,
)
from torch.utils.tensorboard import SummaryWriter


CfgLike = Union[str, Mapping[str, Any]]


def prepare_dinov3_cfg(
    cfg: CfgLike, *, backbone_type: str | None = None
) -> dict[str, Any]:
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
        student_cfg["pos_embed_rope_normalize_coords"] = (
            "separate" if rope_norm else "min"
        )
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


def create_tensorboard_writer(
    log_dir: str | Path | None, rank: int, enabled: bool = True
) -> SummaryWriter | None:
    if not enabled or rank != 0:
        return None
    default_dir = Path("runs")
    log_root = Path(log_dir).expanduser() if log_dir is not None else default_dir
    log_root.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_root))


def tensorboard_log_scalars(
    writer: SummaryWriter | None, step: int, scalars: dict[str, float]
) -> None:
    if writer is None:
        return
    for key, value in scalars.items():
        writer.add_scalar(key, value, step)
    writer.flush()


def _tensor_to_uint8_image(
    image_tensor: Optional[torch.Tensor],
    mean_tensor: torch.Tensor,
    std_tensor: torch.Tensor,
):
    if image_tensor is None:
        return None
    image = image_tensor.detach().clone()
    if image.ndim == 4:
        image = image.squeeze(0)
    mean_broadcast = mean_tensor.to(image.dtype)
    std_broadcast = std_tensor.to(image.dtype)
    image = image * std_broadcast + mean_broadcast
    image = image.clamp(0, 255).to(torch.uint8)
    image = image.permute(1, 2, 0).cpu().numpy()
    return image
