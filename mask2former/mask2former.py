import torch

from transformers import Mask2FormerConfig
from MAS.model.Dino_LoRA import LoRA_qkv
from MAS.mask2former.hf_pixel_decoder_custom import (
    DinoCompatibleMask2FormerPixelDecoder,
)
from MAS.mask2former.cddfuse_depth_prior_add import CDDFuseDepthPriorAdd
from typing import Sequence
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerTransformerModule,
)


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
        self.cddfuse_depth_prior_add = None
        if bool(getattr(config, "use_cddfuse_depth_prior_add", False)):
            self.cddfuse_depth_prior_add = CDDFuseDepthPriorAdd(
                embed_dim=int(embed_dim),
                feature_order=self.feature_order,
                image_mean=getattr(config, "image_normalize_mean", None),
                image_std=getattr(config, "image_normalize_std", None),
                prior_channels=int(getattr(config, "cddfuse_prior_channels", 64)),
                gamma_init=float(getattr(config, "cddfuse_gamma", 0.1)),
                fusion_loss_weight=float(
                    getattr(config, "cddfuse_fusion_loss_weight", 1.0)
                ),
                decomp_loss_weight=float(
                    getattr(config, "cddfuse_decomp_loss_weight", 2.0)
                ),
                fusion_size=int(getattr(config, "cddfuse_fusion_size", 256)),
            )
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
        self.adapter.requires_grad_(True)
        self.backbone.requires_grad_(False)
        for module in self.backbone.modules():
            if isinstance(module, LoRA_qkv):
                module.set_lora_requires_grad(True)
        self._logged_feature_info = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
        depth_maps: torch.Tensor | None = None,
    ):
        features_dict = self.adapter(pixel_values)
        if self.cddfuse_depth_prior_add is not None:
            features_dict = self.cddfuse_depth_prior_add(
                pixel_values,
                depth_maps,
                features_dict,
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
            if self.cddfuse_depth_prior_add is not None:
                gamma = ", ".join(
                    f"{value:.4f}"
                    for value in self.cddfuse_depth_prior_add.gamma.detach()
                    .float()
                    .cpu()
                    .tolist()
                )
                print(
                    "[CDDFuseDepthPriorAdd] enabled=True "
                    f"prior_channels={self.cddfuse_depth_prior_add.pyramid[0].block[0].out_channels} "
                    f"gamma=[{gamma}] fusion_loss_weight={self.cddfuse_depth_prior_add.fusion_loss_weight:.4f} "
                    f"decomp_loss_weight={self.cddfuse_depth_prior_add.decomp_loss_weight:.4f} "
                    f"fusion_size={self.cddfuse_depth_prior_add.fusion_size}"
                )
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
            if self.cddfuse_depth_prior_add is not None:
                aux_losses = self.cddfuse_depth_prior_add.auxiliary_losses()
                raw_aux_losses = self.cddfuse_depth_prior_add.raw_losses()
                for key, value in aux_losses.items():
                    loss_dict[key] = value
                    raw_loss_dict[key] = value.detach()
                for key, value in raw_aux_losses.items():
                    raw_loss_dict[key] = value.detach()
                    loss_dict[key] = value.detach()
                if aux_losses:
                    loss = loss + sum(aux_losses.values())
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
