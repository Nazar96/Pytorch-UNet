""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_size(f, w=None, s=1):
    if s == 1:
        res = (f - 1)/2
    else:
        res = (w * s - w - s + f) / 2
    res = int(res)
    return res


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, k_size=3, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=k_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=k_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class GridUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        k_size = 5
        p_size = pad_size(k_size)
        self.h_conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=p_size)
        self.v_conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=p_size)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=p_size)
        self.final_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=k_size, padding=p_size)

        # self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, 2)

        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.v_pool = nn.AdaptiveAvgPool2d((None, 1))

        self.batch_1 = nn.BatchNorm2d(out_channels * 4)
        self.batch_2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def forward(self, x1, x2):
        h, w = x2.shape[-2:]

        x1 = self.up(x1)
        v_x = self.v_conv(x1)
        h_x = self.h_conv(x1)
        mask_x = self.mask_conv(x1)

        v_x = self.v_pool(v_x).repeat(1, 1, 1, w)
        h_x = self.h_pool(h_x).repeat(1, 1, h, 1)

        x = torch.cat([mask_x, h_x, v_x, x2], 1)
        x = self.batch_1(x)
        x = self.act(x)

        x = self.final_conv(x)
        x = self.batch_2(x)
        x = self.act(x)

        return x
