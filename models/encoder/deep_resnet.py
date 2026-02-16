import torch
import torch.nn as nn
from typing import List
from .resnet import BasicBlock

class DeepResNetTokens(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        stem_ch: int = 32,
        stages: List[int] = (2, 2, 2, 2),  # 4 stages instead of 3
        strides: List[int] = (2, 2, 2, 2),  # 4 strides instead of 3
        channels: List[int] = (32, 64, 128, 256),  # Additional channel size
        norm: str = "bn",
        emb_dim: int = 256,
        freeze_backbone: bool = False,
        return_map: bool = False,
        gn_groups: int = 32,
    ):
        super().__init__()
        assert len(stages) == len(strides) == len(channels)

        self.return_map = return_map
        self.emb_dim = emb_dim
        self.latent_ndim = 2
        self.use_cls_token = False

        # --- Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, stride=1, padding=1, bias=False),
            (
                nn.BatchNorm2d(stem_ch)
                if norm == "bn"
                else nn.GroupNorm(gn_groups, stem_ch)
            ),
            nn.ReLU(inplace=True),
        )

        in_c = stem_ch
        self.stages = nn.ModuleList()
        for num_blocks, out_c, s in zip(stages, channels, strides):
            blocks = []
            if num_blocks > 0:
                blocks.append(
                    BasicBlock(
                        in_c, out_c, stride=s, norm=norm, gn_groups=gn_groups
                    )
                )
                in_c = out_c
                for _ in range(num_blocks - 1):
                    blocks.append(
                        BasicBlock(
                            in_c,
                            out_c,
                            stride=1,
                            norm=norm,
                            gn_groups=gn_groups,
                        )
                    )
            self.stages.append(nn.Sequential(*blocks))

        self.out_c = in_c

        # --- Compute spatial size from strides
        from math import prod
        total_ds = prod(strides) if len(strides) > 0 else 1
        hw = 64 // total_ds
        assert 64 % total_ds == 0, "64 must be divisible by product of strides"

        self.out_hw = hw
        self.patch_size = 64 // self.out_hw

        # --- 1×1 projection to emb_dim
        self.proj = nn.Conv2d(self.out_c, emb_dim, 1)
        self._init_weights()

        if freeze_backbone:
            for m in [self.stem, *self.stages, self.proj]:
                for p in m.parameters():
                    p.requires_grad = False

        self.emb_dim = emb_dim

    def _init_weights(self):
        # Kaiming init for convs; zeros for Norms' bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Slightly smaller std for 1x1 projection for stability
        nn.init.kaiming_uniform_(self.proj.weight, a=1.0)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 3, 64, 64]
        returns:
            tokens: [B, H*W, D]  (H=W=4 for 4-stage network)
            (optional) fmap: [B, D, H, W]
        """
        assert (
            x.shape[-1] == 64 and x.shape[-2] == 64
        ), "This module is sized for 64x64 inputs."

        x = self.stem(x)  # [B, C, 64, 64]
        for stage in self.stages:  # → [B, C, 4, 4] with 4 stages
            if len(stage) > 0:
                x = stage(x)
        x = self.proj(x)  # [B, D, out_hw, out_hw]
        B, D, H, W = x.shape
        assert (
            H == self.out_hw and W == self.out_hw
        ), "Spatial size drifted from metadata"
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, D]
        return (tokens, x) if self.return_map else tokens