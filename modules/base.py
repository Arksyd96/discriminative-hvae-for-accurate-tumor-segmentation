import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimePositionalEmbedding(nn.Module):
    def __init__(self, dimension, T=1000, local_device=None) -> None:
        super().__init__()
        self.embedding = torch.zeros(T, dimension)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension, 2).float() * (-np.log(10000.0) / dimension))
        self.embedding[:, 0::2] = torch.sin(position * div_term)
        self.embedding[:, 1::2] = torch.cos(position * div_term)
        if local_device is not None:
            self.embedding = self.embedding.to(local_device)
    
    def forward(self, timestep):
        return self.embedding[timestep]
    
    def to(self, device):
        super().to(device)
        self.embedding = self.embedding.to(device)
        return self
    
class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        mean = self.weight.mean(dim=[1, 2, 3], keepdim=True)
        var = torch.var(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
        normalized_weight = (self.weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8) -> None:
        super().__init__()
        self.conv = WeightStandardizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, pooling=True) -> None:
        super().__init__()
        if pooling:
            self.downsampler = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=scale_factor),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.downsampler = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=scale_factor, padding=1)
    
    def forward(self, x):
        return self.downsampler(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, transposed=False) -> None:
        super().__init__()
        if transposed:
            self.upsampler = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=scale_factor, padding=1)
        else:
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
    
    def forward(self, x):
        return self.upsampler(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim=None, groups=8) -> None:
        super().__init__()
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, out_channels),
        ) if temb_dim is not None else nn.Identity()

        self.block_a = ConvBlock(in_channels, out_channels, groups=groups)
        self.block_b = ConvBlock(out_channels, out_channels, groups=groups)
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb=None):
        h = self.block_a(x)
        if temb is not None:
            h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.block_b(h)
        return h + self.residual_proj(x)
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, head_dim=32, groups=32) -> None:
        super().__init__()
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q = nn.Conv2d(in_channels, num_heads * head_dim, kernel_size=1)
        self.k = nn.Conv2d(in_channels, num_heads * head_dim, kernel_size=1)
        self.v = nn.Conv2d(in_channels, num_heads * head_dim, kernel_size=1)
        self.norm = nn.GroupNorm(groups, in_channels)
        self.proj = nn.Conv2d(num_heads * head_dim, in_channels, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        q = self.q(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = self.k(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.v(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        attention = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        attention = torch.matmul(attention, v)
        attention = attention.permute(0, 1, 3, 2).contiguous().view(B, self.num_heads * self.head_dim, H, W)
        return self.norm(x + self.proj(attention))
    
class FlashSelfAttention(nn.Module): # you need to enable the flash self attention in torch
    def __init__(self,
        in_channels,
        num_heads=8,
        head_dim=32,
        mask=None,
        dropout=0.0,
        groups=32
    ) -> None:
        super().__init__()
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q = nn.Conv2d(in_channels, num_heads * head_dim, kernel_size=1)
        self.k = nn.Conv2d(in_channels, num_heads * head_dim, kernel_size=1)
        self.v = nn.Conv2d(in_channels, num_heads * head_dim, kernel_size=1)
        self.norm = nn.GroupNorm(groups, in_channels)
        self.proj = nn.Conv2d(num_heads * head_dim, in_channels, kernel_size=1)
        self.mask = mask # TODO (process)
        self.dropout = dropout

    def forward(self, x):
        B, _, H, W = x.shape
        q = self.q(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2).contiguous()
        k = self.k(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2).contiguous()
        v = self.v(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2).contiguous()
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask, dropout_p=self.dropout)
        attention = attention.permute(0, 1, 3, 2).contiguous().view(B, self.num_heads * self.head_dim, H, W)
        return self.norm(x + self.proj(attention))
    
class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim=None, downsample=True, attn=False, num_blocks=2, groups=32) -> None:
        super().__init__()

        self.resnet = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, temb_dim=temb_dim, groups=groups)
            for i in range(num_blocks)
        ])

        self.attn = nn.ModuleList([
            SelfAttention(out_channels, num_heads=8, head_dim=32, groups=groups)
            if attn else nn.Identity()
            for _ in range(num_blocks)
        ])

        self.downsample = Downsample(out_channels, out_channels) if downsample else nn.Identity()

    def forward(self, x, temb=None):
        for resnet_block, attn_block in zip(self.resnet, self.attn):
            x = resnet_block(x, temb)
            x = attn_block(x)
        return self.downsample(x)
    
class DecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim=None, upsample=True, attn=False, num_blocks=2, groups=32) -> None:
        super().__init__()
        self.resnet = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, temb_dim=temb_dim, groups=groups)
            for i in range(num_blocks)
        ])
        self.attn = nn.ModuleList([
            SelfAttention(out_channels, num_heads=8, head_dim=32, groups=groups)
            if attn else nn.Identity()
            for _ in range(num_blocks)
        ])
        self.upsample = Upsample(out_channels, out_channels) if upsample else nn.Identity()

    def forward(self, x, temb=None):
        for resnet_block, attn_block in zip(self.resnet, self.attn):
            x = resnet_block(x, temb)
            x = attn_block(x)
        return self.upsample(x)
    