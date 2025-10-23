import yaml
from copy import deepcopy
from types import SimpleNamespace
from collections.abc import Mapping, MutableMapping
from typing import Any, Union


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
