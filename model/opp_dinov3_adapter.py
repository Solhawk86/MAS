from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3.eval.segmentation.models.backbone.dinov3_adapter import (
    DINOv3_Adapter,
    deform_inputs,
)


class DINOv3OPPAdapter(DINOv3_Adapter):
    """DINOv3 adapter with DiveSeg-style object-prior deep supervision."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = self.backbone.embed_dim
        self.opp_heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv2d(
                            embed_dim,
                            1,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                        )
                        for _ in range(3)
                    ]
                )
                for _ in range(len(self.interaction_indexes))
            ]
        )
        self.opp_heads.apply(self._init_weights)

    def _split_multiscale_tokens(self, c, c2_len, c3_len, bs, dim, h_c, w_c):
        c2 = c[:, 0:c2_len, :]
        c3 = c[:, c2_len : c2_len + c3_len, :]
        c4 = c[:, c2_len + c3_len :, :]
        c2 = c2.transpose(1, 2).view(bs, dim, h_c * 2, w_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, h_c, w_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, h_c // 2, w_c // 2).contiguous()
        return c2, c3, c4

    def _flatten_multiscale_features(self, c2, c3, c4):
        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2)
        c4 = c4.flatten(2).transpose(1, 2)
        return torch.cat([c2, c3, c4], dim=1)

    def _predict_opp(self, block_idx, c2, c3, c4, h_c, w_c):
        prior_maps = []
        for head, feat in zip(self.opp_heads[block_idx], (c2, c3, c4)):
            logits = head(feat)
            logits = F.interpolate(logits, size=(h_c, w_c), mode="bilinear", align_corners=False)
            prior_maps.append(logits)
        return torch.cat(prior_maps, dim=1)

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)

        c2_len = c2.size(1)
        c3_len = c3.size(1)
        c = torch.cat([c2, c3, c4], dim=1)

        h_c, w_c = x.shape[2] // 16, x.shape[3] // 16
        h_toks, w_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size

        with torch.autocast("cuda", torch.bfloat16):
            all_layers = self.backbone.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )

        x_for_shape, _ = all_layers[0]
        bs, _, dim = x_for_shape.shape
        del x_for_shape

        outs = []
        opp_outputs = []
        for i, layer in enumerate(self.interactions):
            x, cls = all_layers[i]
            _, c, _ = layer(
                x,
                c,
                cls,
                deform_inputs1,
                deform_inputs2,
                h_c,
                w_c,
                h_toks,
                w_toks,
            )
            cur_c2, cur_c3, cur_c4 = self._split_multiscale_tokens(
                c,
                c2_len,
                c3_len,
                bs,
                dim,
                h_c,
                w_c,
            )
            opp_outputs.append(self._predict_opp(i, cur_c2, cur_c3, cur_c4, h_c, w_c))
            c = self._flatten_multiscale_features(cur_c2, cur_c3, cur_c4)
            outs.append(x.transpose(1, 2).view(bs, dim, h_toks, w_toks).contiguous())

        c2, c3, c4 = self._split_multiscale_tokens(c, c2_len, c3_len, bs, dim, h_c, w_c)
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, size=(4 * h_c, 4 * w_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * h_c, 2 * w_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * h_c, 1 * w_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(h_c // 2, w_c // 2), mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        outputs = {
            "1": self.norm1(c1),
            "2": self.norm2(c2),
            "3": self.norm3(c3),
            "4": self.norm4(c4),
        }
        outputs["_opp_logits"] = torch.cat(opp_outputs, dim=1)
        return outputs
