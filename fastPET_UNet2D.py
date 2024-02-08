"""
2D U-NET IMPLEMENTATION: U-NET ADAPTED FOR DENOISING.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # with torch.no_grad(): IF NECESSARY
        self.conv1.weight = nn.init.kaiming_normal_(self.conv1.weight)
        self.conv1.bias = nn.init.zeros_(self.conv1.bias)
        self.conv2.weight = nn.init.kaiming_normal_(self.conv2.weight)
        self.conv2.bias = nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.AvgPool2d((2, 2))

    def forward(self, inputs):

        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_channels + out_channels, out_channels)

        # with torch.no_grad(): IF NECESSARY
        self.up.weight = nn.init.zeros_(self.up.weight)
        self.up.bias = nn.init.zeros_(self.up.bias)

    def forward(self, inputs, skip):

        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class UNet2D(nn.Module):

    def __init__(self):

        super().__init__()

        """ Encoder """
        self.e1 = EncoderBlock(1, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        """ Bottleneck """
        self.b = ConvBlock(512, 1024)

        """ Decoder """
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):

        """ Encoder """
        x1, p1 = self.e1(inputs)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, x4)
        d2 = self.d2(d1, x3)
        d3 = self.d3(d2, x2)
        d4 = self.d4(d3, x1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs
