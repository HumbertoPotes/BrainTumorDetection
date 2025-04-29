import torch
import torch.nn as nn

class ConvTumorDetector(nn.Module):
    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size-1)//2
            stride = 2

            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
                nn.GroupNorm(1, out_channels),
                nn.ReLU()
            )

            # residual connection
            if in_channels != out_channels:
                self.skip = nn.Conv2d(in_channels, out_channels, 1, 2, 0)
            else:
                self.skip = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.skip(x) + self.model(x)

    class UpBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size-1)//2
            stride = 2
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size, stride, padding, output_padding=1),
                nn.ReLU()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
    ):
        super().__init__()

        h = 640
        w = 640
        out_channels = 16
        down_layers = 4

        # layers = [
        #     torch.nn.Conv2d(in_channels, out_channels, kernel_size=11, stride=2, padding=5),
        #     torch.nn.ReLU(),
        # ]
        layers = []
        # in_channels = 1
        #in_channels = out_channels
        for _ in range(1,down_layers+1):
            out_channels = out_channels * 2
            layers.append(self.DownBlock(in_channels, out_channels))
            in_channels = out_channels
        for _ in range(1,3):
            out_channels = in_channels // 2 
            layers.append(self.UpBlock(in_channels, out_channels))
            in_channels = out_channels // 2
        
        layers.append(torch.nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0))
        self.model = torch.nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shared_out = self.up2(self.up1(self.db4(self.db3(self.db2(self.db1(x))))))
        # return self.logitshead(shared_out)
        return self.model(x).squeeze(1)#.argmax(dim=1).float()


    # def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     logits = self(x)
    #     pred = logits

    #     return pred
