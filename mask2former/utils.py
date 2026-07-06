import torch
import numpy as np
from scipy import ndimage

from typing import Iterable, Sequence
from transformers import Mask2FormerConfig
from dinov3.eval.segmentation.models import BackboneLayersSet


def build_mask2former_config(
    cfg: dict, *, dataset_key: str | None = "dataset"
) -> Mask2FormerConfig:
    dataset_cfg = cfg[dataset_key] if dataset_key is not None else {}
    mask2former_cfg = cfg["mask2former"]
    num_classes = (
        int(dataset_cfg["num_classes"])
        if dataset_cfg
        else int(mask2former_cfg["num_classes"])
    )
    ignore_value = (
        int(dataset_cfg["ignore_index"])
        if dataset_cfg
        else int(mask2former_cfg["ignore_value"])
    )
    hidden_dim = int(mask2former_cfg["hidden_dim"])
    ffn_dim = int(mask2former_cfg["ffn_dim"])
    config = Mask2FormerConfig(
        feature_size=int(mask2former_cfg["feature_size"]),
        mask_feature_size=int(mask2former_cfg["mask_feature_size"]),
        hidden_dim=hidden_dim,
        encoder_feedforward_dim=ffn_dim,
        encoder_layers=int(mask2former_cfg["encoder_layers"]),
        decoder_layers=int(mask2former_cfg["decoder_layers"]),
        num_attention_heads=int(mask2former_cfg["num_heads"]),
        dim_feedforward=ffn_dim,
        dropout=float(mask2former_cfg["dropout"]),
        pre_norm=bool(mask2former_cfg["pre_norm"]),
        enforce_input_projection=bool(mask2former_cfg["enforce_input_projection"]),
        common_stride=int(mask2former_cfg["common_stride"]),
        num_queries=int(mask2former_cfg["num_queries"]),
        no_object_weight=float(mask2former_cfg["no_object_weight"]),
        class_weight=float(mask2former_cfg["class_weight"]),
        mask_weight=float(mask2former_cfg["mask_weight"]),
        dice_weight=float(mask2former_cfg["dice_weight"]),
        train_num_points=int(mask2former_cfg["train_num_points"]),
        oversample_ratio=float(mask2former_cfg["oversample_ratio"]),
        importance_sample_ratio=float(mask2former_cfg["importance_sample_ratio"]),
        feature_strides=list(mask2former_cfg["feature_strides"]),
        num_labels=num_classes,
        ignore_value=ignore_value,
        use_auxiliary_loss=True,
        output_auxiliary_logits=True,
    )

    transforms_cfg = dataset_cfg["transforms"] if dataset_cfg else {}
    mean = transforms_cfg["mean"] if transforms_cfg else None
    std = transforms_cfg["std"] if transforms_cfg else None
    if mean and std and len(mean) == len(std):
        config.image_normalize_mean = list(mean)
        config.image_normalize_std = list(std)

    config.use_cddfuse_depth_prior_add = bool(
        mask2former_cfg.get("use_cddfuse_depth_prior_add", False)
    )
    config.cddfuse_gamma = float(mask2former_cfg.get("cddfuse_gamma", 0.1))
    config.cddfuse_fusion_loss_weight = float(
        mask2former_cfg.get("cddfuse_fusion_loss_weight", 1.0)
    )
    config.cddfuse_decomp_loss_weight = float(
        mask2former_cfg.get("cddfuse_decomp_loss_weight", 2.0)
    )
    config.cddfuse_prior_channels = int(
        mask2former_cfg.get("cddfuse_prior_channels", 64)
    )
    config.cddfuse_fusion_size = int(mask2former_cfg.get("cddfuse_fusion_size", 256))

    return config


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


def _split_foreground_components(
    binary_mask: torch.Tensor,
    *,
    min_pixels: int = 1,
) -> list[torch.Tensor]:
    """
    对前景二值 mask 做连通域拆分，返回每个实例 mask。
    """
    if binary_mask.dtype != torch.bool:
        binary_mask = binary_mask.to(torch.bool)
    array = binary_mask.detach().cpu().numpy().astype(np.uint8)
    labeled, num_components = ndimage.label(array)
    if num_components == 0:
        return []
    components: list[torch.Tensor] = []
    for comp_id in range(1, num_components + 1):
        component = labeled == comp_id
        if not component.any():
            continue
        if min_pixels > 1 and component.sum() < min_pixels:
            continue
        tensor = torch.from_numpy(component.astype(np.float32)).to(binary_mask.device)
        components.append(tensor)
    return components


def _build_targets(
    masks: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    min_instance_pixels: int = 6,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    将语义 mask 拆成 Mask2Former 的实例级标签。
    忽略像素数低于 `min_instance_pixels` 的碎片。
    """
    mask_labels: list[torch.Tensor] = []
    class_labels: list[torch.Tensor] = []
    for mask in masks:
        unique_vals = torch.unique(mask)
        bin_masks: list[torch.Tensor] = []
        cls_ids: list[int] = []
        for value in unique_vals.tolist():
            if value == ignore_index or value < 0 or value >= num_classes:
                continue
            value_int = int(value)
            binary = mask == value_int
            if torch.count_nonzero(binary) == 0:
                continue
            if value_int == 0:
                bin_masks.append(binary.float())
                cls_ids.append(value_int)
                continue
            components = _split_foreground_components(
                binary, min_pixels=max(1, min_instance_pixels)
            )
            if not components:
                continue
            bin_masks.extend(component.float() for component in components)
            cls_ids.extend([value_int] * len(components))
        if bin_masks:
            mask_labels.append(
                torch.stack(bin_masks).to(mask.device, dtype=torch.float32)
            )
            class_labels.append(
                torch.as_tensor(cls_ids, device=mask.device, dtype=torch.long)
            )
        else:
            h, w = mask.shape[-2:]
            mask_labels.append(
                torch.zeros((0, h, w), device=mask.device, dtype=torch.float32)
            )
            class_labels.append(torch.zeros((0,), device=mask.device, dtype=torch.long))
    return mask_labels, class_labels


def _build_targets_without_split(
    masks: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    mask_labels: list[torch.Tensor] = []
    class_labels: list[torch.Tensor] = []
    for mask in masks:
        unique_vals = torch.unique(mask)
        bin_masks: list[torch.Tensor] = []
        cls_ids: list[int] = []
        for value in unique_vals.tolist():
            if value == ignore_index or value < 0 or value >= num_classes:
                continue
            binary = (mask == value).float()
            if torch.count_nonzero(binary) == 0:
                continue
            bin_masks.append(binary)
            cls_ids.append(int(value))
        if bin_masks:
            mask_labels.append(torch.stack(bin_masks))
            class_labels.append(
                torch.as_tensor(cls_ids, device=mask.device, dtype=torch.long)
            )
        else:
            h, w = mask.shape[-2:]
            mask_labels.append(
                torch.zeros((0, h, w), device=mask.device, dtype=torch.float32)
            )
            class_labels.append(torch.zeros((0,), device=mask.device, dtype=torch.long))
    return mask_labels, class_labels
