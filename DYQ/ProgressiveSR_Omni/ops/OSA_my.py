import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from ProgressiveSR_Omni.ops.layernorm import LayerNorm2d
import ProgressiveSR_Omni.ops.esa


# helper classes


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class Conv_PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.):
        super().__init__()

        hidden_features = int(dim*mult)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# MBConv

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


def MBConv(
        dim_in,
        dim_out,
        *,
        downsample,
        expansion_rate=4,
        shrinkage_rate=0.25,
        dropout=0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        # nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes
class Attention(nn.Module):
    """
    这就是普通空间attention
    """

    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            window_size=7,
            with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim ** -0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        # relative positional bias相对位置编码
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, h = *x.shape, self.heads
        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        # add positional bias
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            dots = dots + rearrange(bias, 'i j h -> h i j')

        attn = self.attn(dots)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, '(b x y) h (w1 w2) d -> b x y w1 w2 (h d)',
                        x=height, y=width, w1=window_height, w2=window_width)
        out = self.to_out(out)
        return out


class Channel_Attention(nn.Module):
    """
    通道注意力，转换成CxHW与CxHW进行运算
    """

    def __init__(
            self,
            dim,
            heads,
            bias=False,
            window_size=7
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = rearrange(qkv, 'b (qkv head d) (h ph) (w pw)->qkv (b h w) head d (ph pw)',
                            qkv=3, head=self.heads, ph=self.ps, pw=self.ps)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.temperature
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, '(b h w) head d (ph pw)->b (head d) (h ph) (w pw)',
                        h=h // self.ps, w=w // self.ps, ph=self.ps, pw=self.ps, head=self.heads)
        out = self.project_out(out)
        return out


class Channel_Attention_grid(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias=False,
            window_size=7
    ):
        super(Channel_Attention_grid, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = rearrange(qkv, 'b (qkv head d) (h ph) (w pw)->qkv (b ph pw) head d (h w)',
                            qkv=3, head=self.heads, ph=self.ps, pw=self.ps)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.temperature
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, '(b ph pw) head d (h w)->b (head d) (h ph) (w pw)',
                        h=h // self.ps, w=w // self.ps, ph=self.ps, pw=self.ps, head=self.heads)
        out = self.project_out(out)
        return out


class OSA_Block(nn.Module):
    def __init__(self, channel_num=64, bias=True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0):
        super(OSA_Block, self).__init__()
        w = window_size
        self.layer = nn.Sequential(
            MBConv(
                channel_num,
                channel_num,
                downsample=False,
                expansion_rate=1,
                shrinkage_rate=0.25
            ),
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),
            PreNormResidual(channel_num, Attention(dim=channel_num,
                                                   dim_head=channel_num // 4,
                                                   dropout=dropout,
                                                   window_size=window_size,
                                                   with_pe=with_pe)),
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num)),

            Conv_PreNormResidual(channel_num, Channel_Attention(dim=channel_num,
                                                                heads=4,
                                                                window_size=window_size)),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num)),

            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),
            PreNormResidual(channel_num, Attention(dim=channel_num,
                                                   dim_head=channel_num // 4,
                                                   dropout=dropout,
                                                   window_size=window_size,
                                                   with_pe=with_pe)),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num)),

            Conv_PreNormResidual(channel_num, Channel_Attention_grid(dim=channel_num,
                                                                     heads=4,
                                                                     window_size=window_size)),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num))
        )

    def forward(self, x):
        out = self.layer(x)
        return out
