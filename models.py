"""
Implementation of the different models.

These models are :
    - U-Net for semantic segmentation
    - Siamese Network for template matching
    - Spatial Transformer Network for homography refinement
"""

###########
# Imports #
###########

import torch
import torch.nn as nn
from kornia.geometry.warp import HomographyWarper


###########
# Classes #
###########

# Generic classes

class Conv(nn.Sequential):
    """Generic convolution layer"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class LConv(nn.Sequential):
    """Generic convolution layer"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )


class DoubleConv(nn.Sequential):
    """Generic double convolution layer"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            Conv(in_channels, out_channels, kernel_size, stride, padding),
            Conv(out_channels, out_channels, kernel_size, stride, padding)
        )


class Deconv(nn.Sequential):
    """Generic deconvolution layer"""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Dense(nn.Sequential):
    """Generic dense layer"""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )


# Models

class UNet(nn.Module):
    """
    Implementation of a U-Net network for semantic
    segmentation of field images.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = DoubleConv(in_channels, 32)
        self.conv2 = DoubleConv(32, 64)
        self.conv3 = DoubleConv(64, 128)
        self.conv4 = DoubleConv(128, 128)
        self.conv5 = DoubleConv(128 + 128, 128)
        self.conv6 = DoubleConv(64 + 64, 64)
        self.conv7 = DoubleConv(32 + 32, 32)

        self.upsample1 = Deconv(128, 128)
        self.upsample2 = Deconv(128, 64)
        self.upsample3 = Deconv(64, 32)

        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)

        self.last = nn.Sequential(
            nn.Conv2d(32, out_channels, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Downhill
        d1 = self.conv1(x)
        x = self.maxpool(d1)

        d2 = self.conv2(x)
        x = self.maxpool(d2)

        d3 = self.conv3(x)
        x = self.maxpool(d3)

        x = self.conv4(x)

        # Uphill
        x = self.upsample1(x)
        x = torch.cat([x[:, :, :d3.shape[-2:][0], :d3.shape[-2:][1]], d3], dim=1)
        x = self.conv5(x)

        x = self.upsample2(x)
        x = torch.cat([x[:, :, :d2.shape[-2:][0], :d2.shape[-2:][1]], d2], dim=1)
        x = self.conv6(x)

        x = self.upsample3(x)
        x = torch.cat([x[:, :, :d1.shape[-2:][0], :d1.shape[-2:][1]], d1], dim=1)
        x = self.conv7(x)

        return self.last(x)


class Siamese(nn.Module):
    """
    Implementation of a Siamese Network to find the template
    that best matches a semantic image of a field.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            DoubleConv(in_channels, 8),
            nn.MaxPool2d(4),

            DoubleConv(8, 16),
            nn.MaxPool2d(2),

            DoubleConv(16, 32),
            nn.MaxPool2d(2)
        )

        self.dense = nn.Sequential(
            Dense(4608, 2048),
            Dense(2048, 512)
        )

        self.last = nn.Linear(512, out_channels)

    def encode(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)

        return self.last(x)

    def forward(self, x0, x1):
        x0 = self.encode(x0)
        x1 = self.encode(x1)

        return x0, x1


class STN(nn.Module):
    """
    Implementation of a Spatial Transformer Network to refine
    homographies.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = LConv(in_channels, 16)
        self.conv2 = LConv(16, 16)
        self.conv3 = LConv(16, 32)
        self.conv4 = LConv(32, 32)
        self.conv5 = LConv(32, 64)
        self.conv6 = LConv(64, 64)

        self.pool1 = nn.MaxPool2d(4)
        self.pool2 = nn.MaxPool2d(2)

        self.dense = nn.Sequential(
            Dense(9216, 4096),
            Dense(4096, 1024)
        )

        self.last = nn.Linear(1024, out_channels)

        # Initialize such that all elements in the kernel are zero
        self.last.weight.data.zero_()

        # Initialize such that the bias is to the first 8 values of a flattened identity matrix
        self.last.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float))

        self.warper = HomographyWarper(144, 256, mode='nearest')

    def forward(self, x):
        # Get the relative transformation theta
        d1 = self.conv1(x)
        x = self.conv2(d1)

        x = self.conv2(x + d1)
        x = self.pool1(x)

        d2 = self.conv3(x)
        x = self.conv4(d2)

        x = self.conv4(x + d2)
        x = self.pool2(x)

        d3 = self.conv5(x)
        x = self.conv6(d3)

        x = self.conv6(x + d3)
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)

        theta = self.dense(x)
        theta = self.last(theta)

        return theta


