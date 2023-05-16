from torch import nn


class Net(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        # Input for MNIST is [B, 1, 28, 28]
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # [B, 16, 28, 28]
            nn.ReLU(inplace=True),  # [B, 16, 28, 28]
            nn.MaxPool2d(2, 2),  # [B, 16, 14, 14]
            # =================================================================
            nn.Conv2d(16, 32, 3, padding=1),  # [B, 32, 14, 14]
            nn.ReLU(inplace=True),  # [B, 32, 14, 14]
            nn.MaxPool2d(2, 2),  # [B, 32, 7, 7]
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),  # [B, 64]
            nn.ReLU(inplace=True),  # [B, 64]
            nn.Linear(64, n_classes),  # [B, n_classes]
        )

    def forward(self, img):
        x = self.convolutions(img)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fully_connected(x)

        return x
