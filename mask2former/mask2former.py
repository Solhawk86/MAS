import torch

from transformers import Mask2FormerConfig
from MAS.model.Dino_LoRA import LoRA_qkv
from MAS.mask2former.hf_pixel_decoder_custom import (
    DinoCompatibleMask2FormerPixelDecoder,
)
from typing import Sequence
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerTransformerModule,
)

from MAS.loss import LaplacianClarity, QueryDepthConsistencyLoss
from MAS.mask2former.depth_transformer import DepthAwareMask2FormerTransformerModule
from MAS.mask2former.fg_bg_cosine import ForegroundBackgroundPrototypeCosineLoss


class DinoMask2Former(torch.nn.Module):
    def __init__(
        self,
        adapter: torch.nn.Module,
        config: Mask2FormerConfig,
        feature_order: Sequence[str],
    ):
        super().__init__()
        self.adapter = adapter
        self.feature_order = list(feature_order)
        self.backbone = getattr(adapter, "backbone", None)
        if self.backbone is None:
            raise ValueError("DinoMask2Former 需要显式提供带 backbone 的 adapter")
        embed_dim = getattr(self.backbone, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("无法从 DINOv3 主干中推断 embed_dim")
        self.pixel_decoder = DinoCompatibleMask2FormerPixelDecoder(
            config,
            feature_channels=[int(embed_dim)] * len(self.feature_order),
        )
        self.use_depth_embedding = bool(getattr(config, "use_depth_embedding", True))
        if self.use_depth_embedding:
            self.transformer_module = DepthAwareMask2FormerTransformerModule(
                in_features=config.feature_size,
                config=config,
            )
        else:
            self.transformer_module = Mask2FormerTransformerModule(
                in_features=config.feature_size,
                config=config,
            )
        self.class_predictor = torch.nn.Linear(config.hidden_dim, config.num_labels + 1)
        self.loss_weights = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }
        self.criterion = Mask2FormerLoss(
            config=config,
            weight_dict=self.loss_weights,
        )
        self.config = config
        self.use_fg_bg_cosine_loss = bool(
            getattr(config, "use_fg_bg_cosine_loss", False)
        )
        self.fg_bg_cosine_weight = float(
            getattr(config, "fg_bg_cosine_weight", 0.05)
        )
        self.fg_bg_cosine_loss = None
        if self.use_fg_bg_cosine_loss:
            self.fg_bg_cosine_loss = ForegroundBackgroundPrototypeCosineLoss(
                feature_keys=list(
                    getattr(config, "fg_bg_cosine_feature_keys", ["2", "3", "4"])
                ),
                margin=float(getattr(config, "fg_bg_cosine_margin", 0.0)),
            )
        self.adapter.requires_grad_(True)
        self.backbone.requires_grad_(False)
        for module in self.backbone.modules():
            if isinstance(module, LoRA_qkv):
                module.set_lora_requires_grad(True)
        self._logged_feature_info = False

        # depth-aware queries 的额外监督配置
        self.depth_loss_weight = float(getattr(config, "depth_loss_weight"))
        depth_groups = int(getattr(config, "depth_num_groups"))
        depth_div_weight = float(getattr(config, "depth_diversity_weight"))
        self.depth_loss_enabled = (
            self.use_depth_embedding
            and self.depth_loss_weight > 0
            and depth_groups > 0
        )
        if self.depth_loss_enabled:
            self.depth_clarity = LaplacianClarity()
            self.depth_consistency_loss = QueryDepthConsistencyLoss(
                num_queries=config.num_queries,
                hidden_dim=config.hidden_dim,
                num_depth=depth_groups,
                div_loss_weight=depth_div_weight,
            )
        else:
            self.depth_clarity = None
            self.depth_consistency_loss = None

        mean = getattr(config, "image_normalize_mean", None)
        std = getattr(config, "image_normalize_std", None)
        self._image_mean: torch.Tensor | None = None
        self._image_std: torch.Tensor | None = None
        self._image_divisor = 255.0  # 恢复 0~255 归一化的尺度
        self._images_are_255 = False
        if mean and std and len(mean) == len(std):
            self._image_mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
            self._image_std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
            self._images_are_255 = any(abs(m) > 1.0 for m in mean) or any(
                abs(s) > 1.0 for s in std
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
    ):
        features_dict = self.adapter(pixel_values)
        fg_bg_cosine_raw = None
        if (
            self.use_fg_bg_cosine_loss
            and self.fg_bg_cosine_loss is not None
            and mask_labels is not None
            and class_labels is not None
        ):
            fg_bg_cosine_raw = self.fg_bg_cosine_loss(
                features_dict,
                mask_labels,
                class_labels,
            )
        features = [
            features_dict[key] for key in self.feature_order if key in features_dict
        ]
        if not self._logged_feature_info:
            info = ", ".join(
                f"{key}:{tuple(features_dict[key].shape)}"
                for key in self.feature_order
                if key in features_dict
            )
            print(f"[DINOv3 Adapter Features] {info}")
            self._logged_feature_info = True
        decoder_out = self.pixel_decoder(
            features,
            output_hidden_states=True,
            output_attentions=self.config.output_attentions,
            return_dict=True,
        )
        transformer_out = self.transformer_module(
            multi_scale_features=list(decoder_out.multi_scale_features),
            mask_features=decoder_out.mask_features,
            output_hidden_states=True,
            output_attentions=self.config.output_attentions,
        )
        class_queries = tuple(
            self.class_predictor(hidden_state.transpose(0, 1))
            for hidden_state in transformer_out.intermediate_hidden_states
        )
        masks_queries = transformer_out.masks_queries_logits
        aux = [
            {
                "class_queries_logits": cls,
                "masks_queries_logits": mask,
            }
            for cls, mask in zip(class_queries[:-1], masks_queries[:-1])
        ]
        loss = None
        loss_dict: dict[str, torch.Tensor] = {}
        raw_loss_dict: dict[str, torch.Tensor] = {}
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.criterion(
                masks_queries_logits=masks_queries[-1],
                class_queries_logits=class_queries[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=aux,
            )
            raw_loss_dict = {k: v.detach() for k, v in loss_dict.items()}
            loss_dict = self._weight_detection_losses(loss_dict)
            loss = sum(loss_dict.values())

            if fg_bg_cosine_raw is not None:
                raw_loss_dict["loss_fg_bg_cosine"] = fg_bg_cosine_raw.detach()
                weighted_fg_bg_cosine = (
                    fg_bg_cosine_raw * self.fg_bg_cosine_weight
                )
                loss = loss + weighted_fg_bg_cosine
                loss_dict["loss_fg_bg_cosine"] = weighted_fg_bg_cosine

            if self.depth_loss_enabled:
                depth_loss_raw = self._compute_depth_loss(
                    pixel_values,
                    masks_queries[-1],
                    transformer_out.get("depth_latent"),
                    transformer_out.get("depth_ids"),
                )
                if depth_loss_raw is not None:
                    raw_loss_dict["depth_loss"] = depth_loss_raw.detach()
                    weighted_depth = depth_loss_raw * self.depth_loss_weight
                    loss = loss + weighted_depth
                    loss_dict["depth_loss"] = weighted_depth
        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "raw_loss_dict": raw_loss_dict,
            "class_queries_logits": class_queries[-1],
            "masks_queries_logits": masks_queries[-1],
        }

    def _weight_detection_losses(
        self, raw_losses: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """将 HF loss 名称映射到 config 中的权重。"""
        weighted: dict[str, torch.Tensor] = {}
        for key, value in raw_losses.items():
            coeff = next(
                (
                    base_weight
                    for base_key, base_weight in self.loss_weights.items()
                    if key.startswith(base_key)
                ),
                1.0,
            )
            weighted[key] = value * coeff
        return weighted

    def _recover_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        images = pixel_values.detach()
        if self._image_mean is not None and self._image_std is not None:
            # 反归一化回原图分布
            mean = self._image_mean.to(images.device)
            std = self._image_std.to(images.device)
            images = images * std + mean
        if self._images_are_255:
            images = images / self._image_divisor
        return images.clamp(0.0, 1.0)

    def _compute_depth_loss(
        self,
        pixel_values: torch.Tensor,
        mask_logits: torch.Tensor,
        depth_latent: torch.Tensor | None,
        depth_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if (
            not self.depth_loss_enabled
            or self.depth_consistency_loss is None
            or self.depth_clarity is None
            or depth_latent is None
            or depth_ids is None
        ):
            return None
        clarity_map = self.depth_clarity(self._recover_images(pixel_values))
        mask_probs = mask_logits.sigmoid()
        return self.depth_consistency_loss(
            depth_latent=depth_latent,
            masks=mask_probs,
            clarity_map=clarity_map,
            depth_ids=depth_ids,
        )
