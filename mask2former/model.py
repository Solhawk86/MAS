
import torch

from transformers import Mask2FormerConfig
from MAS.mask2former.hf_pixel_decoder_custom import DinoCompatibleMask2FormerPixelDecoder
from typing import Sequence
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerTransformerModule,
)

class DinoMask2Former(torch.nn.Module):
    def __init__(self, feature_extractor: torch.nn.Module, config: Mask2FormerConfig, feature_order: Sequence[str]):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_order = list(feature_order)
        backbone = getattr(feature_extractor, "backbone", feature_extractor)
        embed_dim = getattr(backbone, "embed_dim", None)
        if embed_dim is None:
            raise ValueError("无法从 DINOv3 主干中推断 embed_dim")
        self.pixel_decoder = DinoCompatibleMask2FormerPixelDecoder(
            config,
            feature_channels=[int(embed_dim)] * len(self.feature_order),
        )
        self.transformer_module = Mask2FormerTransformerModule(in_features=config.feature_size, config=config)
        self.class_predictor = torch.nn.Linear(config.hidden_dim, config.num_labels + 1)
        self.criterion = Mask2FormerLoss(
            config=config,
            weight_dict={
                "loss_cross_entropy": config.class_weight,
                "loss_mask": config.mask_weight,
                "loss_dice": config.dice_weight,
            },
        )
        self.config = config
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad_(False)
        self._logged_feature_info = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor] | None = None,
        class_labels: list[torch.Tensor] | None = None,
    ):
        with torch.no_grad():
            features_dict = self.feature_extractor(pixel_values)
        features = [features_dict[key] for key in self.feature_order if key in features_dict]
        if not self._logged_feature_info:
            info = ", ".join(f"{key}:{tuple(features_dict[key].shape)}" for key in self.feature_order if key in features_dict)
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
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.criterion(
                masks_queries_logits=masks_queries[-1],
                class_queries_logits=class_queries[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=aux,
            )
            loss = sum(loss_dict.values())
        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "class_queries_logits": class_queries[-1],
            "masks_queries_logits": masks_queries[-1],
        }