"""Custom Mask2Former pixel decoder that adapts multi-resolution DINOv3 features to HuggingFace expectations."""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelDecoder, Mask2FormerConfig


class DinoCompatibleMask2FormerPixelDecoder(Mask2FormerPixelDecoder):
    """
    Wraps HuggingFace's Mask2FormerPixelDecoder but normalizes the incoming feature list so that
    the transformer branch always receives levels aligned with config.feature_strides.
    """

    def forward(self, features, *args, **kwargs):
        aligned = self._align_features(features)
        return super().forward(aligned, *args, **kwargs)

    def _align_features(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if not features:
            return list(features)

        sorted_feats = sorted(features, key=lambda x: x.shape[-1], reverse=True)
        base_feat = sorted_feats[0]

        def _as_int(value):
            if isinstance(value, torch.Tensor):
                return int(value.item())
            return int(value)

        base_h, base_w = (_as_int(dim) for dim in base_feat.shape[-2:])
        base_stride = self.config.feature_strides[0] if self.config.feature_strides else 4

        stride_to_feature: dict[int, torch.Tensor] = {}
        available = list(sorted_feats)

        for stride in self.config.feature_strides:
            if not available:
                break
            target_h = max(1, int(round(float(base_h * base_stride / stride))))
            target_w = max(1, int(round(float(base_w * base_stride / stride))))

            def _score(idx: int) -> int:
                h = _as_int(available[idx].shape[-2])
                w = _as_int(available[idx].shape[-1])
                return abs(h - target_h) + abs(w - target_w)

            best_idx = min(range(len(available)), key=_score)
            feat = available.pop(best_idx)
            if feat.shape[-2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            stride_to_feature[stride] = feat

        ordered: list[torch.Tensor] = []
        for stride in self.config.feature_strides:
            feat = stride_to_feature.get(stride)
            if feat is not None:
                ordered.append(feat)

        ordered.extend(available)
        return ordered
