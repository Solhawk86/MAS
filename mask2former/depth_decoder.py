from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    ACT2FN,
    Mask2FormerAttention,
    Mask2FormerConfig,
    Mask2FormerMaskedAttentionDecoderOutput,
    Mask2FormerMaskPredictor,
    Mask2FormerSinePositionEmbedding,
)


class DepthTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        token_grid: int = 8,
        token_channels: int = 64,
        use_edge: bool = True,
        use_valid: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.token_grid = int(token_grid)
        self.use_edge = bool(use_edge)
        self.use_valid = bool(use_valid)
        in_channels = 1 + int(self.use_edge) + int(self.use_valid)
        mid_channels = int(token_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
            persistent=False,
        )

    def forward(
        self,
        depth_maps: torch.Tensor,
        depth_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        depth = normalize_depth_map(depth_maps, depth_valid)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        features = [depth]
        if self.use_edge:
            edge = depth_edge_map(depth)
            features.append(edge)
        if self.use_valid:
            if depth_valid is None:
                valid = torch.ones_like(depth)
            else:
                valid = depth_valid.float()
                if valid.ndim == 3:
                    valid = valid.unsqueeze(1)
            features.append(valid)
        x = torch.cat(features, dim=1).to(dtype=depth.dtype)
        x = self.stem(x)
        x = F.adaptive_avg_pool2d(x, (self.token_grid, self.token_grid))
        token_map = self.proj(x.flatten(2).transpose(1, 2))
        tokens = token_map.transpose(0, 1).contiguous()
        diagnostics = {
            "depth_decoder/token_mean": token_map.detach().mean(),
            "depth_decoder/token_std": token_map.detach().std(unbiased=False),
        }
        return tokens, diagnostics


class DepthCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        config: Mask2FormerConfig,
        *,
        gamma_mode: str = "fixed",
        gamma_init: float = 1.0,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_dim
        self.pre_norm = config.pre_norm
        self.self_attn = Mask2FormerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            is_decoder=True,
        )
        depth_heads = int(
            getattr(config, "depth_cross_attention_heads", config.num_attention_heads)
        )
        self.cross_attn = nn.MultiheadAttention(
            self.embed_dim,
            config.num_attention_heads,
            config.dropout,
        )
        self.depth_attn = nn.MultiheadAttention(
            self.embed_dim,
            depth_heads,
            dropout=float(getattr(config, "depth_cross_attention_dropout", config.dropout)),
            batch_first=False,
        )
        self.dropout = config.dropout
        self.depth_dropout = float(
            getattr(config, "depth_cross_attention_dropout", config.dropout)
        )
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.dim_feedforward)
        self.fc2 = nn.Linear(config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        if gamma_mode == "learnable":
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        else:
            self.register_buffer("gamma", torch.tensor(float(gamma_init)), persistent=False)
        self.use_depth_ff_adapter = bool(getattr(config, "use_depth_ff_adapter", False))
        self.depth_ff_adapter_dropout = float(
            getattr(config, "depth_ff_adapter_dropout", 0.0)
        )
        self.depth_ff_adapter_bottleneck_dim = int(
            getattr(
                config,
                "depth_ff_adapter_bottleneck_dim",
                max(1, self.embed_dim // 4),
            )
        )
        if self.use_depth_ff_adapter:
            self.ff_adapter_down = nn.Linear(
                self.embed_dim, self.depth_ff_adapter_bottleneck_dim
            )
            self.ff_adapter_up = nn.Linear(
                self.depth_ff_adapter_bottleneck_dim, self.embed_dim
            )
            self.ff_adapter_activation = nn.GELU()
            ff_gamma_mode = str(
                getattr(config, "depth_ff_adapter_gamma_mode", "learnable")
            )
            ff_gamma_init = float(getattr(config, "depth_ff_adapter_gamma", 0.1))
            if ff_gamma_mode == "learnable":
                self.ff_gamma = nn.Parameter(torch.tensor(ff_gamma_init))
            else:
                self.register_buffer(
                    "ff_gamma", torch.tensor(ff_gamma_init), persistent=False
                )

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        level_index: int,
        position_embeddings: Optional[torch.Tensor],
        query_position_embeddings: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
        depth_tokens: Optional[torch.Tensor],
        depth_token_pos: Optional[torch.Tensor],
        output_attentions: bool = False,
    ):
        cross_attn_weights = None
        self_attn_weights = None

        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.cross_attn_layer_norm(hidden_states)
        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )
        hidden_states = residual + F.dropout(hidden_states, p=self.dropout, training=self.training)
        if not self.pre_norm:
            hidden_states = self.cross_attn_layer_norm(hidden_states)

        residual = hidden_states
        branch_input = (
            self.self_attn_layer_norm(hidden_states) if self.pre_norm else hidden_states
        )
        self_hidden = branch_input
        self_hidden, self_attn_weights = self.self_attn(
            hidden_states=self_hidden,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )
        self_hidden = F.dropout(self_hidden, p=self.dropout, training=self.training)

        if depth_tokens is not None:
            depth_out, depth_attn_weights = self.depth_attn(
                query=self.with_pos_embed(branch_input, query_position_embeddings),
                key=self.with_pos_embed(depth_tokens, depth_token_pos),
                value=depth_tokens,
                need_weights=output_attentions,
            )
            depth_out = F.dropout(depth_out, p=self.depth_dropout, training=self.training)
            scaled_depth = self.gamma * depth_out
        else:
            depth_attn_weights = None
            scaled_depth = None
        hidden_states = residual + self_hidden
        if scaled_depth is not None:
            hidden_states = hidden_states + scaled_depth
        if not self.pre_norm:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        ff = self.final_layer_norm(hidden_states) if self.pre_norm else hidden_states
        ff = self.activation_fn(self.fc1(ff))
        ff = F.dropout(ff, p=self.activation_dropout, training=self.training)
        ff = self.fc2(ff)
        ff = F.dropout(ff, p=self.dropout, training=self.training)
        scaled_adapter = None
        if self.use_depth_ff_adapter:
            adapter = self.ff_adapter_down(residual)
            adapter = self.ff_adapter_activation(adapter)
            adapter = self.ff_adapter_up(adapter)
            adapter = F.dropout(
                adapter, p=self.depth_ff_adapter_dropout, training=self.training
            )
            scaled_adapter = self.ff_gamma * adapter
        hidden_states = residual + ff
        if scaled_adapter is not None:
            hidden_states = hidden_states + scaled_adapter
        if not self.pre_norm:
            hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        diagnostics = {
            "delta_abs_mean": scaled_depth.detach().abs().mean()
            if scaled_depth is not None
            else torch.zeros((), device=hidden_states.device),
            "gamma": self.gamma.detach()
            if torch.is_tensor(self.gamma)
            else torch.tensor(float(self.gamma), device=hidden_states.device),
            "ff_adapter_abs_mean": scaled_adapter.detach().abs().mean()
            if scaled_adapter is not None
            else torch.zeros((), device=hidden_states.device),
            "ff_adapter_gamma": self.ff_gamma.detach()
            if self.use_depth_ff_adapter and torch.is_tensor(self.ff_gamma)
            else torch.zeros((), device=hidden_states.device),
        }
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights, depth_attn_weights)
        outputs += (diagnostics,)
        return outputs


class DepthAwareMask2FormerTransformerModule(nn.Module):
    def __init__(self, in_features: int, config: Mask2FormerConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.num_feature_levels = 3
        self.layerdrop = config.dropout
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.queries_features = nn.Embedding(config.num_queries, hidden_dim)
        self.input_projections = nn.ModuleList()
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_projection:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())
        self.decoder = nn.ModuleList(
            [
                DepthCrossAttentionLayer(
                    config,
                    gamma_mode=str(getattr(config, "depth_cross_attention_gamma_mode", "fixed")),
                    gamma_init=float(getattr(config, "depth_cross_attention_gamma", 1.0)),
                )
                for _ in range(config.decoder_layers - 1)
            ]
        )
        self.layernorm = nn.LayerNorm(config.hidden_dim)
        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=config.hidden_dim,
            num_heads=config.num_attention_heads,
            mask_feature_size=config.mask_feature_size,
        )
        self.depth_token_encoder = DepthTokenEncoder(
            hidden_dim=config.hidden_dim,
            token_grid=int(getattr(config, "depth_token_grid", 8)),
            token_channels=int(getattr(config, "depth_token_channels", 64)),
            use_edge=bool(getattr(config, "depth_token_use_edge", True)),
            use_valid=bool(getattr(config, "depth_token_use_valid", True)),
        )
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        multi_scale_features: List[torch.Tensor],
        mask_features: torch.Tensor,
        depth_maps: torch.Tensor | None = None,
        depth_valid: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []
        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(
                self._position_embedding(multi_scale_features[i]).flatten(2)
            )
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)
        _, batch_size, _ = multi_stage_features[0].shape
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_features = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)
        depth_tokens = None
        depth_token_pos = None
        depth_diagnostics: dict[str, torch.Tensor] = {}
        if depth_maps is not None:
            depth_tokens, depth_diagnostics = self.depth_token_encoder(depth_maps, depth_valid)
            depth_token_pos = torch.zeros_like(depth_tokens)
        hidden_states = query_features
        intermediate = ()
        all_hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        intermediate_mask_predictions = ()
        intermediate_hidden_states = self.layernorm(query_features)
        intermediate += (intermediate_hidden_states,)
        predicted_mask, attention_mask = self.mask_predictor(
            intermediate_hidden_states, mask_features, size_list[0]
        )
        intermediate_mask_predictions += (predicted_mask,)
        for idx, decoder_layer in enumerate(self.decoder):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = torch.rand([], device=hidden_states.device)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            level_index = idx % self.num_feature_levels
            attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False
            layer_outputs = decoder_layer(
                hidden_states,
                level_index=level_index,
                position_embeddings=multi_stage_positional_embeddings,
                query_position_embeddings=query_embeddings,
                encoder_hidden_states=multi_stage_features,
                encoder_attention_mask=attention_mask,
                depth_tokens=depth_tokens,
                depth_token_pos=depth_token_pos,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            layer_diagnostics = layer_outputs[-1]
            for key, value in layer_diagnostics.items():
                depth_diagnostics[f"depth_decoder/layer{idx + 1}_{key}"] = value
            intermediate_hidden_states = self.layernorm(hidden_states)
            predicted_mask, attention_mask = self.mask_predictor(
                intermediate_hidden_states,
                mask_features,
                size_list[(idx + 1) % self.num_feature_levels],
            )
            intermediate_mask_predictions += (predicted_mask,)
            intermediate += (intermediate_hidden_states,)
            if output_attentions:
                attentions += (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        hidden_states = hidden_states.transpose(1, 0)
        if not return_dict:
            outputs = [hidden_states, all_hidden_states, attentions, intermediate, intermediate_mask_predictions]
            return tuple(v for v in outputs if v is not None)
        output = Mask2FormerMaskedAttentionDecoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=attentions,
            masks_queries_logits=intermediate_mask_predictions,
            intermediate_hidden_states=intermediate,
        )
        output.depth_diagnostics = depth_diagnostics
        return output

    def _position_embedding(self, feature: torch.Tensor) -> torch.Tensor:
        try:
            return self.position_embedder(feature.shape, feature.device, feature.dtype)
        except TypeError:
            try:
                return self.position_embedder(feature, None, feature.dtype)
            except TypeError:
                return self.position_embedder(feature, None)


def normalize_depth_map(
    depth_maps: torch.Tensor,
    depth_valid: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    depth = depth_maps.float()
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    valid = torch.isfinite(depth)
    if depth_valid is not None:
        v = depth_valid.float()
        if v.ndim == 3:
            v = v.unsqueeze(1)
        valid = valid & (v > 0.5)
    safe = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.zeros_like(safe)
    for b in range(safe.shape[0]):
        mask = valid[b]
        if mask.any():
            vals = safe[b][mask]
            mean = vals.mean()
            mn = vals.min()
            mx = vals.max()
            filled = torch.where(mask, safe[b], mean)
            out[b] = (filled - mn) / (mx - mn + eps)
    return out.clamp(0.0, 1.0)


def depth_edge_map(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=depth.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=depth.device,
    ).view(1, 1, 3, 3)
    d = depth.float()
    gx = F.conv2d(d, sobel_x, padding=1)
    gy = F.conv2d(d, sobel_y, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + eps)
    denom = edge.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (edge / denom).clamp(0.0, 1.0).to(dtype=depth.dtype)
