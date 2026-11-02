import torch
import torch.nn as nn


class VGGMini(nn.Module):
    """VGG-like CNN à 4 couches conv pour CIFAR-10 (32×32).

    Architecture :
        conv(3→32) → ReLU → conv(32→64) → ReLU → MaxPool(2)   [16×16]
        conv(64→128) → ReLU → conv(128→128) → ReLU → MaxPool(2) [8×8]
        FC(128*8*8→256) → ReLU → Dropout → FC(256→10)

    Couches conv accessibles via named_modules() :
        features.0   Conv2d(3,   32,  3)
        features.2   Conv2d(32,  64,  3)
        features.5   Conv2d(64,  128, 3)
        features.7   Conv2d(128, 128, 3)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),     # 0
            nn.ReLU(inplace=True),              # 1
            nn.Conv2d(32, 64, 3, padding=1),    # 2
            nn.ReLU(inplace=True),              # 3
            nn.MaxPool2d(2, 2),                 # 4  → 16×16
            nn.Conv2d(64, 128, 3, padding=1),   # 5
            nn.ReLU(inplace=True),              # 6
            nn.Conv2d(128, 128, 3, padding=1),  # 7
            nn.ReLU(inplace=True),              # 8
            nn.MaxPool2d(2, 2),                 # 9  → 8×8
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
