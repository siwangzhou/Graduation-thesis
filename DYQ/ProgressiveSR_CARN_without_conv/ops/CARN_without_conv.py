import torch
import torch.nn as nn
from einops import rearrange
from ProgressiveSR_CARN_without_conv.ops.pixelshuffle import pixelshuffle_block
from ProgressiveSR_CARN_without_conv.ops.Quantization import Quantization_RS
from ProgressiveSR_CARN_without_conv import ops
import ProgressiveSR_CARN_without_conv.ops.CARN_common

class CS_DownSample_x4(nn.Module):
    def __init__(self):
        super(CS_DownSample_x4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8*8*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=8, a2=8, c=3)

        return x


class CS_DownSample_xDiff_v4(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(9*9-8*8)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v7(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v7, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(10*10-9*9)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v9(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v9, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(11*11-10*10)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v11(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v11, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(12*12-11*11)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v13(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v13, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(13*13-12*12)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v15(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v15, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(14*14-13*13)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v17(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v17, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(15*15-14*14)*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_UpSample_x1_06(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_06, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16*16*3, kernel_size=15, stride=15, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x

class CS_UpSample_x1_14(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_14, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16*16*3, kernel_size=14, stride=14, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x1_23(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_23, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16*16*3, kernel_size=13, stride=13, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x1_33(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_33, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16*16*3, kernel_size=12, stride=12, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x1_45(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_45, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16*16*3, kernel_size=11, stride=11, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x1_6(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16*16*3, kernel_size=10, stride=10, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x1_77(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_77, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16*16*3, kernel_size=9, stride=9, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x2(nn.Module):
    def __init__(self):
        super(CS_UpSample_x2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16*16*3, kernel_size=8, stride=8, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

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


# a0
class LR_SR_x4_v2_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v2_quant, self).__init__()
        self.layer1 = CS_DownSample_x4()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x2()
        self.layer4 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        LR_expand = self.layer3(LR_processed)
        HR = self.layer4(LR_expand)
        return LR, LR_expand, HR


# a1
class LR_SR_x4_v4_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v4_quant, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()
        self.layer1 = CS_DownSample_xDiff_v4()  # 32卷积核的下采样
        self.v4_quant = Quantization_RS()
        self.layer2 = CS_UpSample_x1_77()
        self.layer3 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.v2_quant(self.v2_downsample(x))
        LR = rearrange(LR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=8, a2=8, c=3)

        LR_diff = self.layer1(x)
        LR_diff = self.v4_quant(LR_diff)

        LR_cat = torch.cat((LR, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=9, a2=9, c=3)

        LR_expand = self.layer2(new_LR)
        HR = self.layer3(LR_expand)
        return new_LR, LR_expand, HR


# a2
class LR_SR_x4_v7_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v7_quant, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()

        self.v4_downsample = CS_DownSample_xDiff_v4()
        self.v4_quant = Quantization_RS()

        self.layer1 = CS_DownSample_xDiff_v7()  # 32卷积核的下采样
        self.v7_quant = Quantization_RS()
        self.layer2 = CS_UpSample_x1_6()
        self.layer3 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.v2_quant(self.v2_downsample(x))
        LR = rearrange(LR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=8, a2=8, c=3)

        LR_diff_v4 = self.v4_quant(self.v4_downsample(x))

        LR_cat_v4 = torch.cat((LR, LR_diff_v4), dim=1)

        LR_2 = LR_cat_v4
        LR_diff = self.layer1(x)
        LR_diff = self.v7_quant(LR_diff)

        LR_cat = torch.cat((LR_2, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=10, a2=10, c=3)

        LR_expand = self.layer2(new_LR)
        HR = self.layer3(LR_expand)
        return new_LR, LR_expand, HR


# a3
class LR_SR_x4_v9_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v9_quant, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()

        self.v4_downsample = CS_DownSample_xDiff_v4()
        self.v4_quant = Quantization_RS()

        self.v7_downsample = CS_DownSample_xDiff_v7()
        self.v7_quant = Quantization_RS()

        self.layer1 = CS_DownSample_xDiff_v9()  # 32卷积核的下采样
        self.v9_quant = Quantization_RS()
        self.layer2 = CS_UpSample_x1_45()
        self.layer3 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.v2_quant(self.v2_downsample(x))
        LR = rearrange(LR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=8, a2=8, c=3)

        LR_diff_v4 = self.v4_quant(self.v4_downsample(x))
        LR_cat_v4 = torch.cat((LR, LR_diff_v4), dim=1)
        LR_2 = LR_cat_v4

        LR_diff_v7 = self.v7_quant(self.v7_downsample(x))
        LR_cat_v7 = torch.cat((LR_2, LR_diff_v7), dim=1)
        LR_3 = LR_cat_v7

        LR_diff = self.layer1(x)
        LR_diff = self.v9_quant(LR_diff)

        LR_cat = torch.cat((LR_3, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=11, a2=11, c=3)

        LR_expand = self.layer2(new_LR)
        HR = self.layer3(LR_expand)
        return new_LR, LR_expand, HR


# a4
class LR_SR_x4_v11_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v11_quant, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()

        self.v4_downsample = CS_DownSample_xDiff_v4()
        self.v4_quant = Quantization_RS()

        self.v7_downsample = CS_DownSample_xDiff_v7()
        self.v7_quant = Quantization_RS()

        self.v9_downsample = CS_DownSample_xDiff_v9()
        self.v9_quant = Quantization_RS()

        self.layer1 = CS_DownSample_xDiff_v11()  # 32卷积核的下采样
        self.v11_quant = Quantization_RS()
        self.layer2 = CS_UpSample_x1_33()
        self.layer3 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.v2_quant(self.v2_downsample(x))
        LR = rearrange(LR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=8, a2=8, c=3)

        LR_diff_v4 = self.v4_quant(self.v4_downsample(x))
        LR_cat_v4 = torch.cat((LR, LR_diff_v4), dim=1)
        LR_2 = LR_cat_v4

        LR_diff_v7 = self.v7_quant(self.v7_downsample(x))
        LR_cat_v7 = torch.cat((LR_2, LR_diff_v7), dim=1)
        LR_3 = LR_cat_v7

        LR_diff_v9 = self.v9_quant(self.v9_downsample(x))
        LR_cat_v9 = torch.cat((LR_3, LR_diff_v9), dim=1)
        LR_4 = LR_cat_v9

        LR_diff = self.layer1(x)
        LR_diff = self.v11_quant(LR_diff)

        LR_cat = torch.cat((LR_4, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=12, a2=12, c=3)

        LR_expand = self.layer2(new_LR)
        HR = self.layer3(LR_expand)
        return new_LR, LR_expand, HR


# a5
class LR_SR_x4_v13_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v13_quant, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()

        self.v4_downsample = CS_DownSample_xDiff_v4()
        self.v4_quant = Quantization_RS()

        self.v7_downsample = CS_DownSample_xDiff_v7()
        self.v7_quant = Quantization_RS()

        self.v9_downsample = CS_DownSample_xDiff_v9()
        self.v9_quant = Quantization_RS()

        self.v11_downsample = CS_DownSample_xDiff_v11()
        self.v11_quant = Quantization_RS()

        self.layer1 = CS_DownSample_xDiff_v13()  # 32卷积核的下采样
        self.v13_quant = Quantization_RS()
        self.layer2 = CS_UpSample_x1_23()
        self.layer3 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.v2_quant(self.v2_downsample(x))
        LR = rearrange(LR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=8, a2=8, c=3)

        LR_diff_v4 = self.v4_quant(self.v4_downsample(x))
        LR_cat_v4 = torch.cat((LR, LR_diff_v4), dim=1)
        LR_2 = LR_cat_v4

        LR_diff_v7 = self.v7_quant(self.v7_downsample(x))
        LR_cat_v7 = torch.cat((LR_2, LR_diff_v7), dim=1)
        LR_3 = LR_cat_v7

        LR_diff_v9 = self.v9_quant(self.v9_downsample(x))
        LR_cat_v9 = torch.cat((LR_3, LR_diff_v9), dim=1)
        LR_4 = LR_cat_v9

        LR_diff_v11 = self.v11_quant(self.v11_downsample(x))
        LR_cat_v11 = torch.cat((LR_4, LR_diff_v11), dim=1)
        LR_5 = LR_cat_v11

        LR_diff = self.layer1(x)
        LR_diff = self.v13_quant(LR_diff)

        LR_cat = torch.cat((LR_5, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=13, a2=13, c=3)

        LR_expand = self.layer2(new_LR)
        HR = self.layer3(LR_expand)
        return new_LR, LR_expand, HR


# a6
class LR_SR_x4_v15_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v15_quant, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()

        self.v4_downsample = CS_DownSample_xDiff_v4()
        self.v4_quant = Quantization_RS()

        self.v7_downsample = CS_DownSample_xDiff_v7()
        self.v7_quant = Quantization_RS()

        self.v9_downsample = CS_DownSample_xDiff_v9()
        self.v9_quant = Quantization_RS()

        self.v11_downsample = CS_DownSample_xDiff_v11()
        self.v11_quant = Quantization_RS()

        self.v13_downsample = CS_DownSample_xDiff_v13()
        self.v13_quant = Quantization_RS()

        self.layer1 = CS_DownSample_xDiff_v15()  # 32卷积核的下采样
        self.v15_quant = Quantization_RS()
        self.layer2 = CS_UpSample_x1_14()
        self.layer3 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.v2_quant(self.v2_downsample(x))
        LR = rearrange(LR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=8, a2=8, c=3)

        LR_diff_v4 = self.v4_quant(self.v4_downsample(x))
        LR_cat_v4 = torch.cat((LR, LR_diff_v4), dim=1)
        LR_2 = LR_cat_v4

        LR_diff_v7 = self.v7_quant(self.v7_downsample(x))
        LR_cat_v7 = torch.cat((LR_2, LR_diff_v7), dim=1)
        LR_3 = LR_cat_v7

        LR_diff_v9 = self.v9_quant(self.v9_downsample(x))
        LR_cat_v9 = torch.cat((LR_3, LR_diff_v9), dim=1)
        LR_4 = LR_cat_v9

        LR_diff_v11 = self.v11_quant(self.v11_downsample(x))
        LR_cat_v11 = torch.cat((LR_4, LR_diff_v11), dim=1)
        LR_5 = LR_cat_v11

        LR_diff_v13 = self.v13_quant(self.v13_downsample(x))
        LR_cat_v13 = torch.cat((LR_5, LR_diff_v13), dim=1)
        LR_6 = LR_cat_v13

        LR_diff = self.layer1(x)
        LR_diff = self.v15_quant(LR_diff)

        LR_cat = torch.cat((LR_6, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=14, a2=14, c=3)

        LR_expand = self.layer2(new_LR)
        HR = self.layer3(LR_expand)
        return new_LR, LR_expand, HR


# a7
class LR_SR_x4_v17_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v17_quant, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()

        self.v4_downsample = CS_DownSample_xDiff_v4()
        self.v4_quant = Quantization_RS()

        self.v7_downsample = CS_DownSample_xDiff_v7()
        self.v7_quant = Quantization_RS()

        self.v9_downsample = CS_DownSample_xDiff_v9()
        self.v9_quant = Quantization_RS()

        self.v11_downsample = CS_DownSample_xDiff_v11()
        self.v11_quant = Quantization_RS()

        self.v13_downsample = CS_DownSample_xDiff_v13()
        self.v13_quant = Quantization_RS()

        self.v15_downsample = CS_DownSample_xDiff_v15()
        self.v15_quant = Quantization_RS()

        self.layer1 = CS_DownSample_xDiff_v17()  # 32卷积核的下采样
        self.v17_quant = Quantization_RS()
        self.layer2 = CS_UpSample_x1_06()
        self.layer3 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.v2_quant(self.v2_downsample(x))
        LR = rearrange(LR, 'b c (h a1) (w a2) -> b (c a1 a2) h w', a1=8, a2=8, c=3)

        LR_diff_v4 = self.v4_quant(self.v4_downsample(x))
        LR_cat_v4 = torch.cat((LR, LR_diff_v4), dim=1)
        LR_2 = LR_cat_v4

        LR_diff_v7 = self.v7_quant(self.v7_downsample(x))
        LR_cat_v7 = torch.cat((LR_2, LR_diff_v7), dim=1)
        LR_3 = LR_cat_v7

        LR_diff_v9 = self.v9_quant(self.v9_downsample(x))
        LR_cat_v9 = torch.cat((LR_3, LR_diff_v9), dim=1)
        LR_4 = LR_cat_v9

        LR_diff_v11 = self.v11_quant(self.v11_downsample(x))
        LR_cat_v11 = torch.cat((LR_4, LR_diff_v11), dim=1)
        LR_5 = LR_cat_v11

        LR_diff_v13 = self.v13_quant(self.v13_downsample(x))
        LR_cat_v13 = torch.cat((LR_5, LR_diff_v13), dim=1)
        LR_6 = LR_cat_v13

        LR_diff_v15 = self.v15_quant(self.v15_downsample(x))
        LR_cat_v15 = torch.cat((LR_6, LR_diff_v15), dim=1)
        LR_7 = LR_cat_v15

        LR_diff = self.layer1(x)
        LR_diff = self.v17_quant(LR_diff)

        LR_cat = torch.cat((LR_7, LR_diff), dim=1)
        new_LR = rearrange(LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=15, a2=15, c=3)

        LR_expand = self.layer2(new_LR)
        HR = self.layer3(LR_expand)
        return new_LR, LR_expand, HR
        