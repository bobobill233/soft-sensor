import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class Shift_channel_mix(nn.Module):
    def __init__(self, shift_size=1):
        super(Shift_channel_mix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):  # x的张量 [B,C,H,W]
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        x1 = torch.roll(x1, self.shift_size, dims=2)  # [:,:,1:,:]

        x2 = torch.roll(x2, -self.shift_size, dims=2)  # [:,:,:-1,:]

        x3 = torch.roll(x3, self.shift_size, dims=3)  # [:,:,:,1:]

        x4 = torch.roll(x4, -self.shift_size, dims=3)  # [:,:,:,:-1]

        x = torch.cat([x1, x2, x3, x4], 1)

        return x

class CSAM(nn.Module):
    def __init__(self, in_channels, att_channels=16, lk_size=13, sk_size=3, reduction=2):
        super().__init__()
        self.in_channels = in_channels
        self.att_channels = att_channels
        self.idt_channels = in_channels - att_channels
        self.lk_size = lk_size
        self.sk_size = sk_size

        # 动态卷积核生成器
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(att_channels, att_channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(att_channels // reduction, att_channels * sk_size * sk_size, 1)
        )
        nn.init.zeros_(self.kernel_gen[-1].weight)
        nn.init.zeros_(self.kernel_gen[-1].bias)

        # 共享静态大核卷积核：定义为参数，非卷积层
        self.lk_filter = nn.Parameter(torch.randn(att_channels, att_channels, lk_size, lk_size))
        nn.init.kaiming_normal_(self.lk_filter, mode='fan_out', nonlinearity='relu')

        # 融合层
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scm =  Shift_channel_mix()
    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.att_channels + self.idt_channels, f"Input channel {C} must match att + idt ({self.att_channels} + {self.idt_channels})"

        # 通道拆分
        F_att, F_idt = torch.split(x, [self.att_channels, self.idt_channels], dim=1)

        # 生成动态卷积核 [B * att, 1, 3, 3]
        kernel = self.kernel_gen(F_att).reshape(B * self.att_channels, 1, self.sk_size, self.sk_size)

        # 动态卷积操作
        F_att_re = rearrange(F_att, 'b c h w -> 1 (b c) h w')
        out_dk = F.conv2d(F_att_re, kernel, padding=self.sk_size // 2, groups=B * self.att_channels)
        out_dk = rearrange(out_dk, '1 (b c) h w -> b c h w', b=B, c=self.att_channels)

        # 静态大核卷积
        out_lk = F.conv2d(F_att, self.lk_filter, padding=self.lk_size // 2)

        # 融合（两个卷积结果加和）
        out_att = out_lk + out_dk

        # 拼接 F_idt（保留通道）
        out = torch.cat([out_att, F_idt], dim=1)
        out = self.scm(out)

        out = self.fusion(out)
        return out