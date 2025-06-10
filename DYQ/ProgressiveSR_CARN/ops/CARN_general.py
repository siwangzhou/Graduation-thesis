import torch
import torch.nn as nn
from einops import rearrange
# from ProgressiveSR_CARN.ops.pixelshuffle import pixelshuffle_block
from ProgressiveSR_CARN.ops.Quantization import Quantization_RS
from ProgressiveSR_CARN import ops
import ProgressiveSR_CARN.ops.CARN_common


class CS_DownSample_x4(nn.Module):
    def __init__(self):
        super(CS_DownSample_x4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8 * 8 * 3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=8, a2=8, c=3)

        return x


class CS_DownSample_xDiff_v4(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(9 * 9 - 8 * 8) * 3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v7(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v7, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(10 * 10 - 9 * 9) * 3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v9(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v9, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(11 * 11 - 10 * 10) * 3, kernel_size=32, stride=32,
                               padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v11(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v11, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(12 * 12 - 11 * 11) * 3, kernel_size=32, stride=32,
                               padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v13(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v13, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(13 * 13 - 12 * 12) * 3, kernel_size=32, stride=32,
                               padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v15(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v15, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(14 * 14 - 13 * 13) * 3, kernel_size=32, stride=32,
                               padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_DownSample_xDiff_v17(nn.Module):
    def __init__(self):
        super(CS_DownSample_xDiff_v17, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=(15 * 15 - 14 * 14) * 3, kernel_size=32, stride=32,
                               padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


class CS_UpSample_x1_06(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_06, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16 * 16 * 3, kernel_size=15, stride=15, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x1_14(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_14, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16 * 16 * 3, kernel_size=14, stride=14, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x1_23(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_23, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16 * 16 * 3, kernel_size=13, stride=13, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x1_33(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_33, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16 * 16 * 3, kernel_size=12, stride=12, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x1_45(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_45, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16 * 16 * 3, kernel_size=11, stride=11, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x1_6(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16 * 16 * 3, kernel_size=10, stride=10, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x1_77(nn.Module):
    def __init__(self):
        super(CS_UpSample_x1_77, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16 * 16 * 3, kernel_size=9, stride=9, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2 ) h w -> b c (h a1) (w a2)', a1=16, a2=16, c=3)

        return x


class CS_UpSample_x2(nn.Module):
    def __init__(self):
        super(CS_UpSample_x2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16 * 16 * 3, kernel_size=8, stride=8, padding=0)

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


# general CARN
class LR_SR_x4_general(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_general, self).__init__()

        self.v2_downsample = CS_DownSample_x4()
        self.v2_quant = Quantization_RS()
        self.v2_upsample = CS_UpSample_x2()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8 * 8 * 3, kernel_size=8, stride=8, padding=0)

        self.v4_downsample = CS_DownSample_xDiff_v4()
        self.v4_quant = Quantization_RS()
        self.v4_upsample = CS_UpSample_x1_77()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9 * 9 * 3, kernel_size=9, stride=9, padding=0)

        self.v7_downsample = CS_DownSample_xDiff_v7()
        self.v7_quant = Quantization_RS()
        self.v7_upsample = CS_UpSample_x1_6()
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=10 * 10 * 3, kernel_size=10, stride=10, padding=0)

        self.v9_downsample = CS_DownSample_xDiff_v9()
        self.v9_quant = Quantization_RS()
        self.v9_upsample = CS_UpSample_x1_45()
        self.conv4 = nn.Conv2d(in_channels=3, out_channels=11 * 11 * 3, kernel_size=11, stride=11, padding=0)

        self.v11_downsample = CS_DownSample_xDiff_v11()
        self.v11_quant = Quantization_RS()
        self.v11_upsample = CS_UpSample_x1_33()
        self.conv5 = nn.Conv2d(in_channels=3, out_channels=12 * 12 * 3, kernel_size=12, stride=12, padding=0)

        self.v13_downsample = CS_DownSample_xDiff_v13()
        self.v13_quant = Quantization_RS()
        self.v13_upsample = CS_UpSample_x1_23()
        self.conv6 = nn.Conv2d(in_channels=3, out_channels=13 * 13 * 3, kernel_size=13, stride=13, padding=0)

        self.v15_downsample = CS_DownSample_xDiff_v15()
        self.v15_quant = Quantization_RS()
        self.v15_upsample = CS_UpSample_x1_14()
        self.conv7 = nn.Conv2d(in_channels=3, out_channels=14 * 14 * 3, kernel_size=14, stride=14, padding=0)

        self.v17_downsample = CS_DownSample_xDiff_v17()  # 32卷积核的下采样
        self.v17_quant = Quantization_RS()
        self.v17_upsample = CS_UpSample_x1_06()

        self.SRModel = CARN(kwards=kwards)

    def forward(self, x):
        v2_LR = self.v2_downsample(x)
        v2_LR_quant = self.v2_quant(v2_LR)
        v2_expandLR = self.v2_upsample(v2_LR_quant)
        v2_SR = self.SRModel(v2_expandLR)

        v2_LR_feat = self.conv1(v2_LR_quant)
        v4_LR_diff = self.v4_downsample(x)
        v4_LR_diff_quant = self.v4_quant(v4_LR_diff)
        v4_LR_cat = torch.cat((v2_LR_feat, v4_LR_diff_quant), dim=1)
        v4_LR = rearrange(v4_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=9, a2=9, c=3)
        v4_expandLR = self.v4_upsample(v4_LR)
        v4_SR = self.SRModel(v4_expandLR)

        v4_LR_feat = self.conv2(v4_LR)
        v7_LR_diff = self.v7_downsample(x)
        v7_LR_diff_quant = self.v7_quant(v7_LR_diff)
        v7_LR_cat = torch.cat((v4_LR_feat, v7_LR_diff_quant), dim=1)
        v7_LR = rearrange(v7_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=10, a2=10, c=3)
        v7_expandLR = self.v7_upsample(v7_LR)
        v7_SR = self.SRModel(v7_expandLR)

        v7_LR_feat = self.conv3(v7_LR)
        v9_LR_diff = self.v9_downsample(x)
        v9_LR_diff_quant = self.v9_quant(v9_LR_diff)
        v9_LR_cat = torch.cat((v7_LR_feat, v9_LR_diff_quant), dim=1)
        v9_LR = rearrange(v9_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=11, a2=11, c=3)
        v9_expandLR = self.v9_upsample(v9_LR)
        v9_SR = self.SRModel(v9_expandLR)

        v9_LR_feat = self.conv4(v9_LR)
        v11_LR_diff = self.v11_downsample(x)
        v11_LR_diff_quant = self.v11_quant(v11_LR_diff)
        v11_LR_cat = torch.cat((v9_LR_feat, v11_LR_diff_quant), dim=1)
        v11_LR = rearrange(v11_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=12, a2=12, c=3)
        v11_expandLR = self.v11_upsample(v11_LR)
        v11_SR = self.SRModel(v11_expandLR)

        v11_LR_feat = self.conv5(v11_LR)
        v13_LR_diff = self.v13_downsample(x)
        v13_LR_diff_quant = self.v13_quant(v13_LR_diff)
        v13_LR_cat = torch.cat((v11_LR_feat, v13_LR_diff_quant), dim=1)
        v13_LR = rearrange(v13_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=13, a2=13, c=3)
        v13_expandLR = self.v13_upsample(v13_LR)
        v13_SR = self.SRModel(v13_expandLR)

        v13_LR_feat = self.conv6(v13_LR)
        v15_LR_diff = self.v15_downsample(x)
        v15_LR_diff_quant = self.v15_quant(v15_LR_diff)
        v15_LR_cat = torch.cat((v13_LR_feat, v15_LR_diff_quant), dim=1)
        v15_LR = rearrange(v15_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=14, a2=14, c=3)
        v15_expandLR = self.v15_upsample(v15_LR)
        v15_SR = self.SRModel(v15_expandLR)

        v15_LR_feat = self.conv7(v15_LR)
        v17_LR_diff = self.v17_downsample(x)
        v17_LR_diff_quant = self.v17_quant(v17_LR_diff)
        v17_LR_cat = torch.cat((v15_LR_feat, v17_LR_diff_quant), dim=1)
        v17_LR = rearrange(v17_LR_cat, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=15, a2=15, c=3)
        v17_expandLR = self.v17_upsample(v17_LR)
        v17_SR = self.SRModel(v17_expandLR)

        return [v2_SR, v4_SR, v7_SR, v9_SR, v11_SR, v13_SR, v15_SR, v17_SR]
