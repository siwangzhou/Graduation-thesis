import torch
import torch.nn as nn
from einops import rearrange
from ProgressiveSR_CARN_16x16_8x8.ops.pixelshuffle import pixelshuffle_block
from ProgressiveSR_CARN_16x16_8x8.ops.Quantization import Quantization_RS
from ProgressiveSR_CARN_16x16_8x8 import ops
import ProgressiveSR_CARN_16x16_8x8.ops.CARN_common


class CS_DownSample_x4(nn.Module):
    def __init__(self):
        super(CS_DownSample_x4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2 * 2 * 3, kernel_size=8, stride=8, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=2, a2=2, c=3)

        return x


class CS_DownSample_xDiff_v11(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v11, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(3 * 3 - 2 * 2) * 3, kernel_size=8, stride=8, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_UpSample_x1_33(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_33, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4 * 4 * 3, kernel_size=3, stride=3, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=4, a2=4, c=3)

        return x


class CS_UpSample_x2(nn.Module):
    def __init__(self):
        super(CS_UpSample_x2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4 * 4 * 3, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=4, a2=4, c=3)

        return x


class Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = ops.CARN_common.ResidualBlock(64, 64)
        self.b2 = ops.CARN_common.ResidualBlock(64, 64)
        self.b3 = ops.CARN_common.ResidualBlock(64, 64)
        self.c1 = ops.CARN_common.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = ops.CARN_common.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = ops.CARN_common.BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3


class CARN(nn.Module):
    def __init__(self, **kwargs):
        super(CARN, self).__init__()
        kwargs = kwargs['kwards']
        scale = kwargs["scale"]
        self.scale = scale
        # print(scale)
        # multi_scale = kwargs.get("multi_scale")
        # group = kwargs.get("group", 1)

        self.sub_mean = ops.CARN_common.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.CARN_common.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = Block(64, 64)
        self.b2 = Block(64, 64)
        self.b3 = Block(64, 64)
        self.c1 = ops.CARN_common.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = ops.CARN_common.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = ops.CARN_common.BasicBlock(64 * 4, 64, 1, 1, 0)

        self.upsample = ops.CARN_common.UpsampleBlock(64, scale=scale)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out


class LR_SR_x4_v2_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v2_quant, self).__init__()
        self.layer1 = CS_DownSample_x4()
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x2()
        self.layer4 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        LR_expand = self.layer3(LR_processed)
        HR = self.layer4(LR_expand)
        return LR, LR_expand, HR


class LR_SR_x4_v11_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v11_quant, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2 * 2 * 3, kernel_size=2, stride=2, padding=0)
        self.layer1 = CS_DownSample_xDiff_v11()  # 32卷积核的下采样
        self.v11_quant = Quantization_RS()
        self.layer2 = CS_UpSample_x1_33()
        self.layer3 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.v2_quant(self.v2_downsample(x))
        LR = self.conv1(LR)

        LR_diff = self.layer1(x)
        LR_diff = self.v11_quant(LR_diff)

        LR_cat = torch.cat((LR, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=3, a2=3, c=3)

        LR_expand = self.layer2(new_LR)
        HR = self.layer3(LR_expand)
        return new_LR, LR_expand, HR