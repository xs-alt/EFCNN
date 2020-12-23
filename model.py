#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 9, stride=2, padding=4)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv1_1 = nn.Conv2d(16, 16, 9, stride=1, padding=4)
        self.conv1_1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv2_2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv3_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv4_4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv4_4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv5_5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5_5_bn = nn.BatchNorm2d(256)

        self.bottom_conv = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bottom_conv_bn = nn.BatchNorm2d(512)
        self.bottom_conv_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bottom_conv_1_bn = nn.BatchNorm2d(512)

        self.conv1_transpose = nn.ConvTranspose2d(
            512, 256, 3, stride=1, padding=1)
        self.conv1_trans_bn = nn.BatchNorm2d(256)
        self.conv1_1_transpose = nn.ConvTranspose2d(
            256, 256, 3, stride=1, padding=1)
        self.conv1_1_trans_bn = nn.BatchNorm2d(256)

        self.conv2_transpose = nn.ConvTranspose2d(
            256, 128, 3, stride=2, padding=1)
        self.conv2_trans_bn = nn.BatchNorm2d(128)
        self.conv2_2_transpose = nn.ConvTranspose2d(
            128, 128, 3, stride=1, padding=1)
        self.conv2_2_trans_bn = nn.BatchNorm2d(128)

        self.conv3_transpose = nn.ConvTranspose2d(
            128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv3_trans_bn = nn.BatchNorm2d(64)
        self.conv3_3_transpose = nn.ConvTranspose2d(
            64, 64, 3, stride=1, padding=1)
        self.conv3_3_trans_bn = nn.BatchNorm2d(64)

        self.conv4_transpose = nn.ConvTranspose2d(
            64, 32, 5, stride=2, padding=2, output_padding=1)
        self.conv4_transpose_bn = nn.BatchNorm2d(32)
        self.conv4_4_transpose = nn.ConvTranspose2d(
            32, 32, 5, stride=1, padding=2)
        self.conv4_4_transpose_bn = nn.BatchNorm2d(32)

        self.conv5_transpose = nn.ConvTranspose2d(
            32, 16, 9, stride=2, padding=4, output_padding=1)
        self.conv5_transpose_bn = nn.BatchNorm2d(16)
        self.conv5_5_transpose = nn.ConvTranspose2d(
            16, 16, 9, stride=1, padding=4)
        self.conv5_5_transpose_bn = nn.BatchNorm2d(16)

        self.pos_conv = nn.ConvTranspose2d(
            16, 1, 5, stride=2, padding=2, output_padding=1)
        self.pos_conv_bn = nn.BatchNorm2d(1)
        self.cos_conv = nn.ConvTranspose2d(
            16, 1, 5, stride=2, padding=2, output_padding=1)
        self.cos_conv_bn = nn.BatchNorm2d(1)
        self.sin_conv = nn.ConvTranspose2d(
            16, 1, 5, stride=2, padding=2, output_padding=1)
        self.sin_conv_bn = nn.BatchNorm2d(1)
        self.width_conv = nn.ConvTranspose2d(
            16, 1, 5, stride=2, padding=2, output_padding=1)
        self.width_conv_bn = nn.BatchNorm2d(1)

        self.connect_conv_16 = ResidualBlock(16, 16)
        self.connect_conv_16_bn = nn.BatchNorm2d(16)

        self.connect_conv_32 = ResidualBlock(32, 32)
        self.connect_conv_32_bn = nn.BatchNorm2d(32)

        self.connect_conv_64 = ResidualBlock(64, 64)
        self.connect_conv_64_bn = nn.BatchNorm2d(64)

        self.connect_conv_128 = ResidualBlock(128, 128)
        self.connect_conv_128_bn = nn.BatchNorm2d(128)

        self.connect_conv_256 = ResidualBlock(256, 256)
        self.connect_conv_256_bn = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x_16 = x

        x = self.conv2_bn(F.relu(self.conv2(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x_32 = x

        x = self.conv3_bn(F.relu(self.conv3(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x_64 = x

        x = self.conv4_bn(F.relu(self.conv4(x)))
        x = self.conv4_4_bn(F.relu(self.conv4_4(x)))
        x_128 = x

        x = self.conv5_bn(F.relu(self.conv5(x)))
        x = self.conv5_5_bn(F.relu(self.conv5_5(x)))
        x_256 = x

        x = self.bottom_conv_bn(F.relu(self.bottom_conv(x)))
        x = self.bottom_conv_1_bn(F.relu(self.bottom_conv_1(x)))

        x = self.conv1_trans_bn(F.relu(self.conv1_transpose(x)))
        x = self.conv1_1_trans_bn(F.relu(self.conv1_1_transpose(x)))
        x = self.connect_conv_256_bn(F.relu(self.connect_conv_256(x + x_256)))

        x = self.conv2_trans_bn(F.relu(self.conv2_transpose(x)))
        x = self.conv2_2_trans_bn(F.relu(self.conv2_2_transpose(x)))
        x = self.connect_conv_128_bn(F.relu(self.connect_conv_128(x + x_128)))

        x = self.conv3_trans_bn(F.relu(self.conv3_transpose(x)))
        x = self.conv3_3_trans_bn(F.relu(self.conv3_3_transpose(x)))
        x = self.connect_conv_64_bn(F.relu(self.connect_conv_64(x + x_64)))

        x = self.conv4_transpose_bn(F.relu(self.conv4_transpose(x)))
        x = self.conv4_4_transpose_bn(F.relu(self.conv4_4_transpose(x)))
        x = self.connect_conv_32_bn(F.relu(self.connect_conv_32(x + x_32)))

        x = self.conv5_transpose_bn(F.relu(self.conv5_transpose(x)))
        x = self.conv5_5_transpose_bn(F.relu(self.conv5_5_transpose(x)))
        x = self.connect_conv_16_bn(F.relu(self.connect_conv_16(x + x_16)))

        pos = self.pos_conv_bn(self.pos_conv(x))
        sin = self.sin_conv_bn(self.sin_conv(x))
        cos = self.cos_conv_bn(self.cos_conv(x))
        width = self.width_conv_bn(self.width_conv(x))
        return pos, cos, sin, width


if __name__ == '__main__':
    model = Model()
    sim_data = Variable(torch.rand(1, 1, 400, 400))
    pos, cos, sin, width = model(sim_data)
    print(pos.shape)
