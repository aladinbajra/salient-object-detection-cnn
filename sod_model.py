"""Small from-scratch CNNs for saliency masks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_VARIANTS = {
    "baseline": {"family": "encoder_decoder", "channels": (16, 32, 64, 128), "batch_norm": False, "dropout": 0.0},
    "improved": {"family": "unet", "channels": (32, 64, 128, 256), "dropout": 0.20},
    "improved_v1": {"family": "unet", "channels": (32, 64, 128, 256), "dropout": 0.20},
    "improved_v2": {"family": "unet", "channels": (32, 64, 128, 256), "dropout": 0.25},
    "improved_v3": {"family": "deep_unet", "channels": (32, 64, 128, 256), "dropout": 0.30},
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool, dropout: float):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SODEncoderDecoder(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(channels) < 3:
            raise ValueError("model needs at least three encoder stages")

        self.encoders = nn.ModuleList()
        in_channels = 3
        for out_channels in channels:
            self.encoders.append(ConvBlock(in_channels, out_channels, batch_norm, dropout))
            in_channels = out_channels

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, batch_norm, dropout)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        in_channels = channels[-1] * 2
        for out_channels in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(out_channels, out_channels, batch_norm, dropout))
            in_channels = out_channels

        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(nn.Conv2d(channels[0], 1, kernel_size=1), nn.Sigmoid())
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        for encoder in self.encoders:
            x = self.pool(encoder(x))

        x = self.bottleneck(x)

        for upconv, decoder in zip(self.upconvs, self.decoders):
            x = decoder(upconv(x))

        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.head(x)


class UNetSOD(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.20):
        super().__init__()
        self.encoders = nn.ModuleList()
        in_channels = 3
        for out_channels in channels:
            self.encoders.append(ConvBlock(in_channels, out_channels, batch_norm=True, dropout=0.0))
            in_channels = out_channels

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, batch_norm=True, dropout=dropout)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        in_channels = channels[-1] * 2
        for out_channels in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(out_channels * 2, out_channels, batch_norm=True, dropout=0.0))
            in_channels = out_channels

        self.head = nn.Sequential(nn.Conv2d(channels[0], 1, kernel_size=1), nn.Sigmoid())
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))

        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.head(x)


class DeepUNetSOD(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.30):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("improved_v3 expects four encoder stages")

        self.encoders = nn.ModuleList()
        in_channels = 3
        for out_channels in channels:
            self.encoders.append(DoubleConv(in_channels, out_channels))
            in_channels = out_channels

        bottleneck_channels = channels[-1] * 2
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(DoubleConv(channels[-1], bottleneck_channels), nn.Dropout2d(dropout))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        in_channels = bottleneck_channels
        for out_channels in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(out_channels * 2, out_channels))
            in_channels = out_channels

        self.head = nn.Sequential(nn.Conv2d(channels[0], 1, kernel_size=1), nn.Sigmoid())
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))

        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.head(x)


def build_model(variant: str = "baseline") -> nn.Module:
    if variant not in MODEL_VARIANTS:
        valid = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"unknown model variant '{variant}'. valid variants: {valid}")

    config = MODEL_VARIANTS[variant].copy()
    family = config.pop("family")
    if family == "encoder_decoder":
        return SODEncoderDecoder(**config)
    if family == "unet":
        return UNetSOD(**config)
    if family == "deep_unet":
        return DeepUNetSOD(**config)
    raise ValueError(f"unknown model family '{family}'")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    for name in MODEL_VARIANTS:
        model = build_model(name)
        x = torch.randn(2, 3, 128, 128)
        y = model(x)
        print(f"{name}: {count_trainable_parameters(model):,} parameters | {tuple(x.shape)} -> {tuple(y.shape)}")
