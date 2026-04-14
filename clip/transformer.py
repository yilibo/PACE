import torch
from torch import nn

from clip.transformer_blocks_vifi import ResidualAttentionBlock


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        if design_details["vision_block"] != "ResidualAttentionBlock":
            raise NotImplementedError(
                f"Unsupported vision block '{design_details['vision_block']}'. "
                "This repository keeps only the base ResidualAttentionBlock path."
            )
        if design_details["text_block"] != "ResidualAttentionBlock":
            raise NotImplementedError(
                f"Unsupported text block '{design_details['text_block']}'."
            )

        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(
                width,
                heads,
                attn_mask,
                prompts_needed > i,
                text_layer,
                i,
                design_details,
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

    def forward_return_attention(self, x):
        attns_all = []
        for i, block in enumerate(self.resblocks):
            x, attns = block(x, return_attention=True)
            attns_all.append(attns)
        attns_all = torch.stack(attns_all, dim=0)
        return x, attns_all
