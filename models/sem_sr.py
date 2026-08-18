import torch
import torch.nn as nn
import torch.nn.functional as F


class CALayer(nn.Module):
    """
    Channel Attention layer matching the checkpoint structure.
    """

    def __init__(self, channel, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.body = nn.Sequential(
            nn.Conv2d(
                channel,
                channel // reduction,
                kernel_size=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel // reduction,
                channel,
                kernel_size=1,
                padding=0,
                bias=True
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        y = self.avg_pool(x)
        y = self.body(y)

        return x * y


class RCAB(nn.Module):
    """
    Residual Channel Attention Block.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.ca = CALayer(
            channels,
            reduction
        )

    def forward(self, x):

        residual = x

        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.ca(y)

        return residual + y


class SEMSuperResolution(nn.Module):
    """
    Single-channel 2x SEM super-resolution model.

    Input:
        [B, 1, 128, 128]

    Output:
        [B, 1, 256, 256]

    Configuration:
        features = 64
        blocks = 12
    """

    def __init__(
        self,
        features=64,
        blocks=12,
        reduction=16
    ):
        super().__init__()

        self.head = nn.Conv2d(
            1,
            features,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.body = nn.Sequential(
            *[
                RCAB(
                    features,
                    reduction
                )
                for _ in range(blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.up_conv = nn.Conv2d(
            features,
            features * 4,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.up_relu = nn.ReLU(
            inplace=True
        )

        self.refine1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.refine_relu = nn.ReLU(
            inplace=True
        )

        self.refine2 = nn.Conv2d(
            features,
            1,
            kernel_size=3,
            padding=1,
            bias=True
        )

    def forward(self, x):

        # Bicubic base
        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False
        )

        features = self.head(x)

        identity = features

        features = self.body(features)

        features = self.body_conv(
            features
        )

        features = features + identity

        features = self.up_conv(
            features
        )

        features = self.pixel_shuffle(
            features
        )

        features = self.up_relu(
            features
        )

        features = self.refine1(
            features
        )

        features = self.refine_relu(
            features
        )

        correction = self.refine2(
            features
        )

        return base + correction