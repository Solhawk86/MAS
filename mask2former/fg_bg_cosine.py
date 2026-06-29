from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForegroundBackgroundPrototypeCosineLoss(nn.Module):
    def __init__(
        self,
        feature_keys: Sequence[str] = ("2", "3", "4"),
        margin: float = 0.0,
        eps: float = 1e-6,
        min_tokens: int = 2,
    ) -> None:
        super().__init__()
        self.feature_keys = tuple(feature_keys)
        self.margin = float(margin)
        self.eps = float(eps)
        self.min_tokens = int(min_tokens)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        mask_labels: Sequence[torch.Tensor],
        class_labels: Sequence[torch.Tensor],
        return_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fused = self._fuse_features(features)
        targets = self._build_foreground_targets(mask_labels, class_labels)

        total = fused.new_tensor(0.0)
        cos_total = fused.new_tensor(0.0)
        valid = 0
        target_size = fused.shape[-2:]
        for image_idx, target in enumerate(targets):
            target_i = target.unsqueeze(0).to(device=fused.device, dtype=fused.dtype)
            if target_i.shape[-2:] != target_size:
                target_i = F.interpolate(target_i, size=target_size, mode="nearest")
            target_i = target_i.squeeze(0).squeeze(0)

            fg_mask = target_i > 0.5
            bg_mask = target_i == 0
            if (
                fg_mask.sum().item() < self.min_tokens
                or bg_mask.sum().item() < self.min_tokens
            ):
                continue

            tokens = fused[image_idx].permute(1, 2, 0).reshape(-1, fused.shape[1])
            tokens = F.normalize(tokens, p=2, dim=1, eps=self.eps)
            fg_tokens = tokens[fg_mask.reshape(-1)]
            bg_tokens = tokens[bg_mask.reshape(-1)]

            fg_proto = F.normalize(
                fg_tokens.mean(dim=0, keepdim=True), p=2, dim=1, eps=self.eps
            )
            bg_proto = F.normalize(
                bg_tokens.mean(dim=0, keepdim=True), p=2, dim=1, eps=self.eps
            )
            cos = (fg_proto * bg_proto).sum(dim=1)
            total = total + F.relu(cos - self.margin).mean()
            cos_total = cos_total + cos.mean()
            valid += 1

        if valid == 0:
            metrics = {
                "fg_bg_cos": total.detach(),
                "fg_bg_separation": total.detach(),
            }
            if return_metrics:
                return total, metrics
            return total

        loss = total / valid
        mean_cos = cos_total / valid
        metrics = {
            "fg_bg_cos": mean_cos.detach(),
            "fg_bg_separation": (1.0 - mean_cos).detach(),
        }
        if return_metrics:
            return loss, metrics
        return loss

    def _fuse_features(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = [key for key in self.feature_keys if key not in features]
        if missing:
            raise KeyError(f"Missing feature maps for fg/bg cosine loss: {missing}")

        base = features[self.feature_keys[0]]
        target_size = base.shape[-2:]
        aligned = []
        for key in self.feature_keys:
            feat = features[key]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(feat)
        return torch.cat(aligned, dim=1)

    @staticmethod
    def _build_foreground_targets(
        mask_labels: Sequence[torch.Tensor],
        class_labels: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        targets: list[torch.Tensor] = []
        for masks_i, classes_i in zip(mask_labels, class_labels):
            if masks_i.numel() == 0:
                h, w = masks_i.shape[-2:]
                targets.append(masks_i.new_zeros((1, h, w), dtype=torch.float32))
                continue

            foreground = masks_i[classes_i > 0]
            if foreground.numel() == 0:
                h, w = masks_i.shape[-2:]
                target = masks_i.new_zeros((1, h, w), dtype=torch.float32)
            else:
                target = foreground.max(dim=0, keepdim=True)[0].to(dtype=torch.float32)
            targets.append(target)
        return targets
