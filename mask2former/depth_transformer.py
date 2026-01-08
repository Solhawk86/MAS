from __future__ import annotations

from typing import List

from torch import Tensor

from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerConfig,
    Mask2FormerMaskedAttentionDecoderOutput,
    Mask2FormerTransformerModule,
)

from MAS.depth import LearnableDepthQueryEmbedding


class DepthAwareMask2FormerTransformerModule(Mask2FormerTransformerModule):
    """Wrap HF Mask2Former transformer to inject depth-aware query embeddings."""

    def __init__(self, in_features: int, config: Mask2FormerConfig):
        super().__init__(in_features=in_features, config=config)
        num_depth_groups = int(getattr(config, "depth_num_groups", 0) or 0)
        if num_depth_groups <= 0:
            raise ValueError("depth_num_groups 必须为正数才能构建 depth query")
        self.depth_query_embedder = LearnableDepthQueryEmbedding(
            num_queries=config.num_queries,
            hidden_dim=config.hidden_dim,
            num_depth=num_depth_groups,
        )
        # 替换原生 query embed，使 decoder 使用 depth-aware queries
        self.queries_embedder = self.depth_query_embedder

    def forward(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Mask2FormerMaskedAttentionDecoderOutput:
        decoder_output = super().forward(
            multi_scale_features=multi_scale_features,
            mask_features=mask_features,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        decoder_output["depth_latent"] = self.depth_query_embedder.get_depth_latent()
        decoder_output["depth_ids"] = self.depth_query_embedder.depth_ids
        return decoder_output
