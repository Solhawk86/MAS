from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForegroundBackgroundAuxiliaryHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_ratio: float = 0.5,
        dropout: float = 0.1,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden_channels = max(1, int(round(in_channels * hidden_ratio)))
        num_groups = self._resolve_num_groups(hidden_channels)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)

    def forward(
        self,
        mask_features: torch.Tensor,
        mask_labels: Sequence[torch.Tensor],
        class_labels: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.head(mask_features)
        target = self._build_targets(
            mask_labels,
            class_labels,
            target_size=logits.shape[-2:],
            device=logits.device,
        )

        loss_ce = F.cross_entropy(logits, target)
        prob_fg = logits.softmax(dim=1)[:, 1]
        target_fg = target.to(dtype=prob_fg.dtype)
        dims = (1, 2)
        intersection = (prob_fg * target_fg).sum(dim=dims)
        denominator = prob_fg.sum(dim=dims) + target_fg.sum(dim=dims)
        loss_dice = 1.0 - ((2.0 * intersection + self.eps) / (denominator + self.eps))
        loss_dice = loss_dice.mean()
        loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = (pred == target).to(dtype=logits.dtype).mean()
            pred_fg = pred == 1
            target_bool = target == 1
            fg_intersection = (pred_fg & target_bool).sum(dim=dims).to(dtype=logits.dtype)
            fg_union = (pred_fg | target_bool).sum(dim=dims).to(dtype=logits.dtype)
            valid = fg_union > 0
            if valid.any():
                fg_iou = (fg_intersection[valid] / fg_union[valid].clamp_min(1.0)).mean()
            else:
                fg_iou = logits.new_tensor(0.0)

        metrics = {
            "loss_fg_bg_ce": loss_ce.detach(),
            "loss_fg_bg_dice": loss_dice.detach(),
            "fg_bg_aux_acc": correct.detach(),
            "fg_bg_aux_fg_iou": fg_iou.detach(),
        }
        return loss, metrics

    @staticmethod
    def _resolve_num_groups(channels: int) -> int:
        for groups in (32, 16, 8, 4, 2):
            if channels % groups == 0:
                return groups
        return 1

    @staticmethod
    def _build_targets(
        mask_labels: Sequence[torch.Tensor],
        class_labels: Sequence[torch.Tensor],
        target_size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        targets: list[torch.Tensor] = []
        for masks_i, classes_i in zip(mask_labels, class_labels):
            if masks_i.numel() == 0:
                h, w = masks_i.shape[-2:]
                target = masks_i.new_zeros((1, h, w), dtype=torch.float32)
            else:
                foreground = masks_i[classes_i > 0]
                if foreground.numel() == 0:
                    h, w = masks_i.shape[-2:]
                    target = masks_i.new_zeros((1, h, w), dtype=torch.float32)
                else:
                    target = foreground.max(dim=0, keepdim=True)[0].to(dtype=torch.float32)

            target = target.unsqueeze(0).to(device=device, dtype=torch.float32)
            if target.shape[-2:] != target_size:
                target = F.interpolate(target, size=target_size, mode="nearest")
            targets.append(target.squeeze(0).squeeze(0).to(dtype=torch.long))
        return torch.stack(targets, dim=0)
