from typing import List
import torch
import torchvision
import torch.nn as nn


class resnet18(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        unit_norm: bool = False,
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.pretrained = pretrained
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unit_norm = unit_norm

        self.latent_ndim = 1
        self.emb_dim = 512
        self.name = "resnet"

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            # flatten all dimensions to batch, then reshape back at the end
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.normalize(x)
        out = self.resnet(x)
        out = self.flatten(out)
        if self.unit_norm:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], -1)
        out = out.unsqueeze(1)
        return out


class resblock(nn.Module):
    # this implementation assumes square images
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=32):
        super(resblock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample

        padding = int((kernel_size - 1) / 2)

        if resample == "down":
            self.skip = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding),
            )
            self.conv1 = nn.Conv2d(
                input_dim, input_dim, kernel_size, padding=padding, bias=False
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding),
                nn.MaxPool2d(2, stride=2),
            )
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample is None:
            self.skip = nn.Conv2d(input_dim, output_dim, 1)
            self.conv1 = nn.Conv2d(
                input_dim, output_dim, kernel_size, padding=padding, bias=False
            )
            self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size, padding=padding)
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)

        self.leakyrelu1 = nn.LeakyReLU()
        self.leakyrelu2 = nn.LeakyReLU()

    def forward(self, x):
        if (self.input_dim == self.output_dim) and self.resample is None:
            idnty = x
        else:
            idnty = self.skip(x)

        residual = x
        residual = self.conv1(residual)
        residual = self.bn1(residual)
        residual = self.leakyrelu1(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.leakyrelu2(residual)

        return idnty + residual


class SmallResNet(nn.Module):
    def __init__(self, output_dim=512):
        super(SmallResNet, self).__init__()

        self.hw = 224

        # 3x224x224
        self.rb1 = resblock(3, 16, 3, resample="down", hw=self.hw)
        # 16x112x112
        self.rb2 = resblock(16, 32, 3, resample="down", hw=self.hw // 2)
        # 32x56x56
        self.rb3 = resblock(32, 64, 3, resample="down", hw=self.hw // 4)
        # 64x28x28
        self.rb4 = resblock(64, 128, 3, resample="down", hw=self.hw // 8)
        # 128x14x14
        self.rb5 = resblock(128, 512, 3, resample="down", hw=self.hw // 16)
        # 512x7x7
        self.maxpool = nn.MaxPool2d(7)
        # 512x1x1
        self.flat = nn.Flatten()

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            # flatten all dimensions to batch, then reshape back at the end
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.maxpool(x)
        out = x.flatten(start_dim=-3)
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], -1)
        return out


# ---- Basic residual block (ResNet-style, 3x3 + 3x3) ----
class BasicBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        norm: str = "bn",
        gn_groups: int = 32,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3, stride=stride, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 3, stride=1, padding=1, bias=False
        )

        if norm == "bn":
            self.n1 = nn.BatchNorm2d(out_ch)
            self.n2 = nn.BatchNorm2d(out_ch)
            self.nskip = (
                nn.BatchNorm2d(out_ch)
                if (stride != 1 or in_ch != out_ch)
                else nn.Identity()
            )
        else:
            # GroupNorm is more stable with small per-GPU batches
            self.n1 = nn.GroupNorm(gn_groups, out_ch)
            self.n2 = nn.GroupNorm(gn_groups, out_ch)
            self.nskip = (
                nn.GroupNorm(gn_groups, out_ch)
                if (stride != 1 or in_ch != out_ch)
                else nn.Identity()
            )

        self.act = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Conv2d(
                in_ch, out_ch, 1, stride=stride, bias=False
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.n1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.n2(out)

        if self.downsample is not None:
            x = self.downsample(x)
            x = self.nskip(x)

        out = self.act(out + x)
        return out


# ---- Mini-ResNet for 64x64 that returns tokens [B, HW, D] ----
class ResNetSmallTokens(nn.Module):
    """
    A  ResNet for 64x64 inputs that returns a grid of tokens for a ViT/Transformer predictor.

    Args:
        in_ch: input channels (3 for RGB)
        stem_ch: channels after the initial 3x3 conv stem
        stages: number of residual blocks per stage (default [2,2,1,0])
        channels: output channels per stage (default [32, 64, 128])
        norm: "bn" or "gn" (GroupNorm recommended for tiny per-GPU batches)
        emb_dim: projection dimension for tokens (D)
        out_grid: "8x8" (default) or "4x4" (adds an extra stride-2 downsample head)
        freeze_backbone: if True, freeze all conv/Norm in backbone and train only the 1x1 projection
        return_map: if True, also return the [B, D, H, W] map besides tokens
    """

    def __init__(
        self,
        in_ch: int = 3,
        stem_ch: int = 32,
        stages: List[int] = (2, 2, 1, 0),  # ≈ ResNet-10ish
        channels: List[int] = (32, 64, 128),
        norm: str = "bn",
        emb_dim: int = 256,
        out_grid: str = "8x8",
        freeze_backbone: bool = False,
        return_map: bool = False,
        gn_groups: int = 32,
    ):
        super().__init__()
        assert len(stages) >= 3, "Provide at least three stage counts"
        assert len(channels) >= 3, "Provide at least three stage channel sizes"
        assert out_grid in {"8x8", "4x4"}

        self.return_map = return_map
        self.patch_size = 8 if out_grid == "8x8" else 4
        self.emb_dim = emb_dim
        self.latent_ndim = 2
        self.use_cls_token = False # for compatibility with other encoders

        # --- Stem: 3x3 conv, stride=1 (keep 64x64), no maxpool 
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, stride=1, padding=1, bias=False),
            (
                nn.BatchNorm2d(stem_ch)
                if norm == "bn"
                else nn.GroupNorm(gn_groups, stem_ch)
            ),
            nn.ReLU(inplace=True),
        )

        # --- Stages: go 64->32->16->8 with strides [2,2,2] across stage1..3
        strides = [2, 2, 2]  # three downsamples → 8x8 output from 64x64
        in_c = stem_ch
        self.stages = nn.ModuleList()
        for si, (num_blocks, out_c, s) in enumerate(
            zip(stages[:3], channels[:3], strides)
        ):
            blocks = []
            # first block can downsample (stride s), remaining are stride 1
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

        self.out_c = in_c  # channels at 8x8

        # Optional final downsample head to 4x4 if requested
        if out_grid == "4x4":
            self.to_4x4 = BasicBlock(
                self.out_c,
                self.out_c,
                stride=2,
                norm=norm,
                gn_groups=gn_groups,
            )
            out_hw = 4
        else:
            self.to_4x4 = nn.Identity()
            out_hw = 8

        # --- 1x1 projection to emb_dim, produce tokens
        self.proj = nn.Conv2d(self.out_c, emb_dim, 1)

        # Init
        self._init_weights()

        # Optional freezing of backbone (everything except proj)
        if freeze_backbone:
            for m in [self.stem, *self.stages, self.to_4x4]:
                for p in m.parameters():
                    p.requires_grad = False

        self.emb_dim = emb_dim
        self.out_hw = out_hw  # 8 or 4

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
            tokens: [B, H*W, D]  (H=W=8 or 4)
            (optional) fmap: [B, D, H, W]
        """
        assert (
            x.shape[-1] == 64 and x.shape[-2] == 64
        ), "This module is sized for 64x64 inputs."

        x = self.stem(x)  # [B, C, 64, 64]
        for stage in self.stages:  # → [B, C, 8, 8]
            if len(stage) > 0:
                x = stage(x)
        x = self.to_4x4(x)  # optional → [B, C, 4, 4]
        x = self.proj(x)  # [B, D, H, W]

        B, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, D]

        if self.return_map:
            return tokens, x
        return tokens
