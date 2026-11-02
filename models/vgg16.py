import torch
import torch.nn as nn


class VGG16Cifar(nn.Module):
    """VGG16 modifié pour CIFAR-10 (32×32) — 3 blocs, 7 conv, 1 Dense (~7M params).

    Architecture (conforme au schéma "Our Modified VGG16") :

        INPUT (3, 32, 32)
        ┌─ Bloc 1 ──────────────────────────────────┐
        │  Conv1  Conv2d(3→128,   3×3, pad=1)       │
        │  Conv2  Conv2d(128→128, 3×3, pad=1)       │
        │  MaxPool(2)              → (128, 16, 16)  │
        └───────────────────────────────────────────┘
        ┌─ Bloc 2 ──────────────────────────────────┐
        │  Conv3  Conv2d(128→256, 3×3, pad=1)       │
        │  Conv4  Conv2d(256→256, 3×3, pad=1)       │
        │  MaxPool(2)              → (256,  8,  8)  │
        └───────────────────────────────────────────┘
        ┌─ Bloc 3 ──────────────────────────────────┐
        │  Conv5  Conv2d(256→512, 3×3, pad=1)       │
        │  Conv6  Conv2d(512→512, 3×3, pad=1)       │
        │  Conv7  Conv2d(512→512, 3×3, pad=1)       │
        │  MaxPool(2)              → (512,  4,  4)  │
        └───────────────────────────────────────────┘
        Flatten → 512×4×4 = 8192
        Dense   Linear(8192 → num_classes)
        OUTPUT

    Total paramètres : ~7.02 M
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # ── Bloc 1 : 2 conv (3 → 128) ────────────────────────
            nn.Conv2d(3,   128, 3, padding=1), nn.ReLU(inplace=True),  # Conv1
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),  # Conv2
            nn.MaxPool2d(2, 2),                                          # → 16×16

            # ── Bloc 2 : 2 conv (128 → 256) ──────────────────────
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),  # Conv3
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),  # Conv4
            nn.MaxPool2d(2, 2),                                          # → 8×8

            # ── Bloc 3 : 3 conv (256 → 512) ──────────────────────
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),  # Conv5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),  # Conv6
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),  # Conv7
            nn.MaxPool2d(2, 2),                                          # → 4×4
        )
        self.classifier = nn.Linear(512 * 4 * 4, num_classes)          # Dense

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
