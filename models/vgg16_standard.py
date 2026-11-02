import torch
import torch.nn as nn


class VGG16Standard(nn.Module):
    """VGG16 standard adapté CIFAR-10 (32×32) — 10 conv, ~7.7M params, ~285M MACs.

    Architecture :
        Conv(3→64)×2   → MaxPool  → 16×16
        Conv(64→128)×2 → MaxPool  →  8×8
        Conv(128→256)×3 → MaxPool →  4×4
        Conv(256→512)×3 → MaxPool →  2×2
        Linear(512×2×2 → 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   64,  3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,  64,  3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                         # 16×16

            nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                         # 8×8

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                         # 4×4

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                         # 2×2
        )
        self.classifier = nn.Linear(512 * 2 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))
