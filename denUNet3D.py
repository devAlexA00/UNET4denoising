"""
3D denU-NET IMPLEMENTATION: U-NET ADAPTED FOR DENOISING.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect')
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same', padding_mode='reflect')
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
        self.pool = nn.AvgPool3d((2, 2, 2)) # Average pooling was chosen over max pooling, as denoising is not a
        # classification task.
        # Batch normalisation was not employed as the scale of intensities of medical images mustn't be normalised.

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_channels + out_channels, out_channels)

        # with torch.no_grad(): IF NECESSARY
        self.up.weight = nn.init.zeros_(self.up.weight)
        self.up.bias = nn.init.zeros_(self.up.bias)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class denUNet3D(nn.Module):

    def __init__(self, n_blocks=None, n_channels=None):

        if n_blocks is None:
            n_blocks = 3
        if n_channels is None:
            n_channels = 64

        super().__init__()
        self.n_blocks = n_blocks

        if n_blocks == 4:
            """ Encoder """
            self.e1 = EncoderBlock(1, n_channels)
            self.e2 = EncoderBlock(n_channels, n_channels * 2)
            self.e3 = EncoderBlock(n_channels * 2, n_channels * 4)
            self.e4 = EncoderBlock(n_channels * 4, n_channels * 8)

            """ Bottleneck """
            self.b = ConvBlock(n_channels * 8, n_channels * 16)

            """ Decoder """
            self.d1 = DecoderBlock(n_channels * 16, n_channels * 8)
            self.d2 = DecoderBlock(n_channels * 8, n_channels * 4)
            self.d3 = DecoderBlock(n_channels * 4, n_channels * 2)
            self.d4 = DecoderBlock(n_channels * 2, n_channels)

            """ Classifier """
            self.outputs = nn.Conv3d(n_channels, 1, kernel_size=1, padding=0)

        elif n_blocks == 3:
            """ Encoder """
            self.e1 = EncoderBlock(1, n_channels)
            self.e2 = EncoderBlock(n_channels, n_channels * 2)
            self.e3 = EncoderBlock(n_channels * 2, n_channels * 4)

            """ Bottleneck """
            self.b = ConvBlock(n_channels * 4, n_channels * 8)

            """ Decoder """
            self.d1 = DecoderBlock(n_channels * 8, n_channels * 4)
            self.d2 = DecoderBlock(n_channels * 4, n_channels * 2)
            self.d3 = DecoderBlock(n_channels * 2, n_channels)

            """ Classifier """
            self.outputs = nn.Conv3d(n_channels, 1, kernel_size=1, padding=0)

    def forward(self, inputs):

        if self.n_blocks == 4:
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

        elif self.n_blocks == 3:
            """ Encoder """
            x1, p1 = self.e1(inputs)
            x2, p2 = self.e2(p1)
            x3, p3 = self.e3(p2)

            """ Bottleneck """
            b = self.b(p3)

            """ Decoder """
            d1 = self.d1(b, x3)
            d2 = self.d2(d1, x2)
            d3 = self.d3(d2, x1)

            """ Classifier """
            outputs = self.outputs(d3)

        return outputs
