import torch

from typing import Iterable, Sequence
from transformers import Mask2FormerConfig
from dinov3.eval.segmentation.models import BackboneLayersSet

def build_mask2former_config(cfg: dict, *, dataset_key: str | None = "dataset") -> Mask2FormerConfig:
    dataset_cfg = cfg.get(dataset_key, {}) or {}
    mask2former_cfg = cfg.get("mask2former", {}) or {}
    num_classes = int(dataset_cfg.get("num_classes", mask2former_cfg.get("num_classes", 2)))
    ignore_value = int(dataset_cfg.get("ignore_index", mask2former_cfg.get("ignore_value", -1)))
    hidden_dim = int(mask2former_cfg.get("hidden_dim", 256))
    ffn_dim = int(mask2former_cfg.get("ffn_dim", 2048))
    return Mask2FormerConfig(
        feature_size=int(mask2former_cfg.get("feature_size", hidden_dim)),
        mask_feature_size=int(mask2former_cfg.get("mask_feature_size", hidden_dim)),
        hidden_dim=hidden_dim,
        encoder_feedforward_dim=ffn_dim,
        encoder_layers=int(mask2former_cfg.get("encoder_layers", 6)),
        decoder_layers=int(mask2former_cfg.get("decoder_layers", 9)),
        num_attention_heads=int(mask2former_cfg.get("num_heads", 8)),
        dim_feedforward=ffn_dim,
        dropout=float(mask2former_cfg.get("dropout", 0)),
        pre_norm=bool(mask2former_cfg.get("pre_norm", False)),
        enforce_input_projection=bool(mask2former_cfg.get("enforce_input_projection", False)),
        common_stride=int(mask2former_cfg.get("common_stride", 4)),
        num_queries=int(mask2former_cfg.get("num_queries", 100)),
        no_object_weight=float(mask2former_cfg.get("no_object_weight", 0.1)),
        class_weight=float(mask2former_cfg.get("class_weight", 2.0)),
        mask_weight=float(mask2former_cfg.get("mask_weight", 5.0)),
        dice_weight=float(mask2former_cfg.get("dice_weight", 5.0)),
        train_num_points=int(mask2former_cfg.get("train_num_points", 11250)),
        oversample_ratio=float(mask2former_cfg.get("oversample_ratio", 3.0)),
        importance_sample_ratio=float(mask2former_cfg.get("importance_sample_ratio", 0.75)),
        feature_strides=list(mask2former_cfg.get("feature_strides", [4, 8, 16, 32])),
        num_labels=num_classes,
        ignore_value=ignore_value,
        use_auxiliary_loss=True,
        output_auxiliary_logits=True,
    )
    
def resolve_interaction_indices(backbone: torch.nn.Module, cfg: dict) -> list[int]:
    spec = (
        cfg.get("model", {}).get("backbone_out_layers")
        or cfg.get("backbone", {}).get("backbone_out_layers")
        or BackboneLayersSet.FOUR_EVEN_INTERVALS.name
    )
    n_blocks = getattr(backbone, "n_blocks", None)
    if isinstance(spec, str):
        try:
            layerset = BackboneLayersSet[spec]
        except KeyError:
            values = [int(part.strip()) for part in spec.split(",") if part.strip()]
            if not values:
                raise ValueError(f"无法解析 backbone_out_layers = {spec}")
            return values
        if n_blocks is None:
            raise ValueError("backbone 缺少 n_blocks，无法根据枚举推断层索引")
        if layerset == BackboneLayersSet.LAST:
            return [n_blocks - 1]
        if layerset == BackboneLayersSet.FOUR_LAST:
            return list(range(max(0, n_blocks - 4), n_blocks))
        if layerset == BackboneLayersSet.FOUR_EVEN_INTERVALS:
            if n_blocks == 24:
                return [4, 11, 17, 23]
            step = max(1, n_blocks // 4)
            return [min(n_blocks - 1, step * i - 1) for i in range(1, 5)]
        raise ValueError(f"暂不支持的 BackboneLayersSet: {layerset}")
    if isinstance(spec, int):
        if n_blocks is None:
            raise ValueError("整数模式需要 backbone.n_blocks")
        count = max(1, spec)
        return list(range(max(0, n_blocks - count), n_blocks))
    if isinstance(spec, Iterable):
        values = [int(v) for v in spec]
        if not values:
            raise ValueError("backbone_out_layers 不能为空")
        return values
    raise TypeError(f"无法解析的 backbone_out_layers 类型: {type(spec)}")

def _as_dtype(name: str, device_type: str) -> torch.dtype:
    key = (name or "").strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "half", "float16"}:
        if device_type != "cuda":
            raise ValueError("float16 自动混合精度仅支持 CUDA")
        return torch.float16
    if key in {"fp32", "float32", "f32", ""}:
        return torch.float32
    raise ValueError(f"未知的 autocast dtype: {name}")
