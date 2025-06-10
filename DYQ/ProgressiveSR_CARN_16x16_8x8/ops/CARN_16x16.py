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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4 * 4 * 3, kernel_size=16, stride=16, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=4, a2=4, c=3)

        return x


class CS_DownSample_xDiff_v7(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v7, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(5 * 5 - 4 * 4) * 3, kernel_size=16, stride=16, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v11(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v11, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(6 * 6 - 5 * 5) * 3, kernel_size=16, stride=16, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v15(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v15, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(7 * 7 - 6 * 6) * 3, kernel_size=16, stride=16, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_UpSample_x1_14(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_14, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8 * 8 * 3, kernel_size=7, stride=7, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=8, a2=8, c=3)

        return x


class CS_UpSample_x1_33(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_33, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8 * 8 * 3, kernel_size=6, stride=6, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=8, a2=8, c=3)

        return x


class CS_UpSample_x1_6(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8 * 8 * 3, kernel_size=5, stride=5, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=8, a2=8, c=3)

        return x


class CS_UpSample_x2(nn.Module):
    def __init__(self):
        super(CS_UpSample_x2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8 * 8 * 3, kernel_size=4, stride=4, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=8, a2=8, c=3)

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


class LR_SR_x4_v7_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v7_quant, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4 * 4 * 3, kernel_size=4, stride=4, padding=0)
        self.layer1 = CS_DownSample_xDiff_v7()
        self.v7_quant = Quantization_RS()
        self.layer2 = CS_UpSample_x1_6()
        self.layer3 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.v2_quant(self.v2_downsample(x))
        LR = self.conv1(LR)

        LR_diff = self.layer1(x)
        LR_diff = self.v7_quant(LR_diff)

        LR_cat = torch.cat((LR, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=5, a2=5, c=3)

        LR_expand = self.layer2(new_LR)
        HR = self.layer3(LR_expand)
        return new_LR, LR_expand, HR


class LR_SR_x4_v11_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v11_quant, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4 * 4 * 3, kernel_size=4, stride=4, padding=0)
        self.v7_downsample = CS_DownSample_xDiff_v7()
        self.v7_quant = Quantization_RS()

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5 * 5 * 3, kernel_size=5, stride=5, padding=0)

        self.layer1 = CS_DownSample_xDiff_v11()
        self.v11_quant = Quantization_RS()
        self.layer2 = CS_UpSample_x1_33()
        self.layer3 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.v2_quant(self.v2_downsample(x))
        LR = self.conv1(LR)

        LR_diff_v7 = self.v7_quant(self.v7_downsample(x))

        LR_cat_v7 = torch.cat((LR, LR_diff_v7), dim=1)
        new_LR_v7 = rearrange(LR_cat_v7, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=5, a2=5, c=3)

        LR_2 = self.conv2(new_LR_v7)
        LR_diff = self.layer1(x)
        LR_diff = self.v11_quant(LR_diff)

        LR_cat = torch.cat((LR_2, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=6, a2=6, c=3)

        LR_expand = self.layer2(new_LR)
        HR = self.layer3(LR_expand)
        return new_LR, LR_expand, HR


class LR_SR_x4_v15_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v15_quant, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4 * 4 * 3, kernel_size=4, stride=4, padding=0)

        self.v7_downsample = CS_DownSample_xDiff_v7()
        self.v7_quant = Quantization_RS()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5 * 5 * 3, kernel_size=5, stride=5, padding=0)

        self.v11_downsample = CS_DownSample_xDiff_v11()
        self.v11_quant = Quantization_RS()
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=6 * 6 * 3, kernel_size=6, stride=6, padding=0)

        self.layer1 = CS_DownSample_xDiff_v15()
        self.v15_quant = Quantization_RS()
        self.layer2 = CS_UpSample_x1_14()
        self.layer3 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.v2_quant(self.v2_downsample(x))
        LR = self.conv1(LR)

        LR_diff_v7 = self.v7_quant(self.v7_downsample(x))
        LR_cat_v7 = torch.cat((LR, LR_diff_v7), dim=1)
        new_LR_v7 = rearrange(LR_cat_v7, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=5, a2=5, c=3)
        LR_2 = self.conv2(new_LR_v7)

        LR_diff_v11 = self.v11_quant(self.v11_downsample(x))
        LR_cat_v11 = torch.cat((LR_2, LR_diff_v11), dim=1)
        new_LR_v11 = rearrange(LR_cat_v11, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=6, a2=6, c=3)
        LR_3 = self.conv3(new_LR_v11)

        LR_diff = self.layer1(x)
        LR_diff = self.v15_quant(LR_diff)

        LR_cat = torch.cat((LR_3, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=7, a2=7, c=3)

        LR_expand = self.layer2(new_LR)
        HR = self.layer3(LR_expand)
        return new_LR, LR_expand, HR