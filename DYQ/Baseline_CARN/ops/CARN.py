import torch
import torch.nn as nn
from einops import rearrange
from Baseline_CARN.ops.pixelshuffle import pixelshuffle_block
from Baseline_CARN.ops.Quantization import Quantization_RS
from Baseline_CARN import ops
import Baseline_CARN.ops.CARN_common

class CS_DownSample_x4(nn.Module):
    def __init__(self):
        super(CS_DownSample_x4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8*8*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=8, a2=8, c=3)

        return x


class CS_DownSample_x3_55(nn.Module):
    def __init__(self):
        super(CS_DownSample_x3_55, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9*9*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=9, a2=9, c=3)

        return x


class CS_DownSample_x3_2(nn.Module):
    def __init__(self):
        super(CS_DownSample_x3_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10*10*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=10, a2=10, c=3)

        return x


class CS_DownSample_x2_9(nn.Module):
    def __init__(self):
        super(CS_DownSample_x2_9, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=11*11*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=11, a2=11, c=3)

        return x


class CS_DownSample_x2_66(nn.Module):
    def __init__(self):
        super(CS_DownSample_x2_66, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12*12*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=12, a2=12, c=3)

        return x


class CS_DownSample_x2_46(nn.Module):
    def __init__(self):
        super(CS_DownSample_x2_46, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=13*13*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=13, a2=13, c=3)

        return x


class CS_DownSample_x2_28(nn.Module):
    def __init__(self):
        super(CS_DownSample_x2_28, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=14*14*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=14, a2=14, c=3)

        return x


class CS_DownSample_x2_13(nn.Module):
    def __init__(self):
        super(CS_DownSample_x2_13, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=15*15*3, kernel_size=32, stride=32, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = rearrange(x, 'b (c a1 a2) h w -> b c (h a1) (w a2)', a1=15, a2=15, c=3)

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


class LR_SR_x4_v3_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v3_quant, self).__init__()
        self.layer1 = CS_DownSample_x3_55()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x1_77()
        self.layer4 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        LR_expand = self.layer3(LR_processed)
        HR = self.layer4(LR_expand)
        return LR, LR_expand, HR


class LR_SR_x4_v6_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v6_quant, self).__init__()
        self.layer1 = CS_DownSample_x3_2()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x1_6()
        self.layer4 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        LR_expand = self.layer3(LR_processed)
        HR = self.layer4(LR_expand)
        return LR, LR_expand, HR


class LR_SR_x4_v8_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v8_quant, self).__init__()
        self.layer1 = CS_DownSample_x2_9()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x1_45()
        self.layer4 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        LR_expand = self.layer3(LR_processed)
        HR = self.layer4(LR_expand)
        return LR, LR_expand, HR


class LR_SR_x4_v10_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v10_quant, self).__init__()
        self.layer1 = CS_DownSample_x2_66()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x1_33()
        self.layer4 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        LR_expand = self.layer3(LR_processed)
        HR = self.layer4(LR_expand)
        return LR, LR_expand, HR


class LR_SR_x4_v12_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v12_quant, self).__init__()
        self.layer1 = CS_DownSample_x2_46()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x1_23()
        self.layer4 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        LR_expand = self.layer3(LR_processed)
        HR = self.layer4(LR_expand)
        return LR, LR_expand, HR


class LR_SR_x4_v14_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v14_quant, self).__init__()
        self.layer1 = CS_DownSample_x2_28()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x1_14()
        self.layer4 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        LR_expand = self.layer3(LR_processed)
        HR = self.layer4(LR_expand)
        return LR, LR_expand, HR


class LR_SR_x4_v16_quant(nn.Module):
    def __init__(self, kwards):
        super(LR_SR_x4_v16_quant, self).__init__()
        self.layer1 = CS_DownSample_x2_13()  # 32卷积核的下采样
        self.layer2 = Quantization_RS()
        self.layer3 = CS_UpSample_x1_06()
        self.layer4 = CARN(kwards=kwards)

    def forward(self, x):
        LR = self.layer1(x)
        LR_processed = self.layer2(LR)
        LR_expand = self.layer3(LR_processed)
        HR = self.layer4(LR_expand)
        return LR, LR_expand, HR