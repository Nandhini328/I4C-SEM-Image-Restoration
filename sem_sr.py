import torch
import torch.nn as nn
import torch.nn.functional as F


class RCAB(nn.Module):
    """Residual Channel Attention Block used by the submitted SR model."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        reduced = max(1, channels // reduction)

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = y * self.ca(y)

        return residual + y


class SEMSuperResolution(nn.Module):
    """
    Single-channel 2x SEM image super-resolution model.

    Input : [B, 1, H, W]
    Output: [B, 1, 2H, 2W]

    Configuration used for the reported 33.1968 dB / 0.8888 SSIM
    held-out validation result:
      - 64 feature channels
      - 12 RCAB blocks
      - PixelShuffle x2
    """

    def __init__(self, features: int = 64, blocks: int = 12):
        super().__init__()

        self.head = nn.Conv2d(
            1, features, kernel_size=3, padding=1
        )

        self.body = nn.Sequential(
            *[RCAB(features) for _ in range(blocks)]
        )

        self.body_conv = nn.Conv2d(
            features, features, kernel_size=3, padding=1
        )

        self.up_conv = nn.Conv2d(
            features, features * 4, kernel_size=3, padding=1
        )

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.up_relu = nn.ReLU(inplace=True)

        self.refine1 = nn.Conv2d(
            features, features, kernel_size=3, padding=1
        )
        self.refine_relu = nn.ReLU(inplace=True)

        self.refine2 = nn.Conv2d(
            features, 1, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual reconstruction around bicubic interpolation.
        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False,
        )

        features = self.head(x)
        identity = features

        features = self.body(features)
        features = self.body_conv(features)
        features = features + identity

        features = self.up_conv(features)
        features = self.pixel_shuffle(features)
        features = self.up_relu(features)

        features = self.refine1(features)
        features = self.refine_relu(features)

        correction = self.refine2(features)

        return base + correction
