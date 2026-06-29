from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectPriorHead(nn.Module):
    def __init__(self, in_channels: int, feature_keys: Sequence[str]):
        super().__init__()
        self.feature_keys = list(feature_keys)
        self.heads = nn.ModuleDict(
            {key: nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0) for key in self.feature_keys}
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> list[torch.Tensor]:
        return [torch.sigmoid(self.heads[key](features[key])) for key in self.feature_keys if key in features]


class OPPLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, floss_beta: float = 0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.floss_beta = floss_beta
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        mask_labels: Sequence[torch.Tensor],
        class_labels: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if not inputs:
            raise ValueError("OPPLoss received no prior logits")

        targets = self._build_foreground_targets(mask_labels, class_labels)
        total_loss = inputs[0].new_tensor(0.0)
        for logits in inputs:
            total_loss = total_loss + self._loss_single_level(logits, targets)
        return total_loss / len(inputs)

    def _build_foreground_targets(
        self,
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

    def _loss_single_level(self, logits: torch.Tensor, targets: Sequence[torch.Tensor]) -> torch.Tensor:
        level_loss = logits.new_tensor(0.0)
        for image_idx, target in enumerate(targets):
            input_i = logits[image_idx : image_idx + 1]
            target_i = target.unsqueeze(0).to(device=input_i.device, dtype=input_i.dtype)
            if input_i.shape[-2:] != target_i.shape[-2:]:
                input_i = F.interpolate(
                    input_i,
                    size=target_i.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            channel_losses = []
            for channel_idx in range(input_i.shape[1]):
                channel_logits = input_i[:, channel_idx : channel_idx + 1]
                channel_losses.append(
                    self.bce_weight * self._bce_dice_loss(channel_logits, target_i)
                    + self._cross_entropy_loss(channel_logits, target_i)
                    + self.dice_weight * self._f_loss(channel_logits, target_i)
                )
            level_loss = level_loss + torch.stack(channel_losses).mean()
        return level_loss / len(targets)

    def _bce_dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss_fn(inputs, targets)
        inter = (inputs * targets).sum()
        eps = 1e-5
        dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
        return bce + (1 - dice.mean())

    def _cross_entropy_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_loss_fn(inputs, targets)

    def _f_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target = targets.squeeze(1)
        prediction = inputs.squeeze(1)
        eps = 1e-10
        batch_size = prediction.size(0)
        true_positive = (prediction * target).view(batch_size, -1).sum(dim=1)
        denominator = self.floss_beta * target.view(batch_size, -1).sum(dim=1) + prediction.view(batch_size, -1).sum(dim=1)
        fmeasure = (1 + self.floss_beta) * true_positive / (denominator + eps)
        return (1 - fmeasure).mean()
