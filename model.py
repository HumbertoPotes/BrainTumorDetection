import torch
import torch.nn as nn


class ConvTumorDetector(nn.Module):
    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            stride = 2

            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
            )

            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride) if in_channels != out_channels else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.skip(x) + self.model(x)

    class UpBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            stride = 2
            self.model = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    output_padding=1,
                ),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    1,
                    padding,
                ),
                nn.GroupNorm(1, out_channels),
                nn.ReLU()
            )
        
            self.skip = nn.ConvTranspose2d(in_channels, out_channels, 1, stride, output_padding=1) if in_channels != out_channels else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.skip(x) + self.model(x)

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
    ):
        super().__init__()

        h = 640
        w = 640
        out_channels = 32
        down_layers = 4

        # layers = [
        #     torch.nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=2, padding=5),
        #     torch.nn.ReLU(),
        # ]
        # in_channels = out_channels
        layers = []
        for _ in range(0, down_layers):
            layers.append(self.DownBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels = out_channels * 2

        self.network = torch.nn.Sequential(*layers)

        layers = []

        for _ in range(0, down_layers - 1):
            out_channels = in_channels // 2
            layers.append(self.UpBlock(in_channels, out_channels))
            in_channels = out_channels

        self.segmentation_head = torch.nn.Sequential(
            *layers, torch.nn.ConvTranspose2d(in_channels, num_classes, 1, stride=2, output_padding=1)
        )
        self.category_head = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_channels * h * w // (2 ** (down_layers+1)), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x)
        return self.segmentation_head(features).squeeze(1), self.category_head(
            features
        ).squeeze(1)
