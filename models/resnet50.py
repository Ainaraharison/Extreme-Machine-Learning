import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """Bloc résiduel bottleneck 3 couches conv (ResNet-50/101/152)."""

    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1):
        super().__init__()
        out_channels = planes * self.expansion

        self.conv1 = nn.Conv2d(in_channels, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_channels, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet50Cifar(nn.Module):
    """ResNet-50 adapté pour CIFAR-10 (entrée 32×32).

    Différences vs ResNet-50 ImageNet :
      - conv1 : kernel 3×3, stride 1, padding 1  (au lieu de 7×7/2)
      - pas de MaxPool2d après conv1
      - GlobalAvgPool2d en fin → compatible 32×32

    Architecture :
        INPUT (3, 32, 32)
        conv1  Conv2d(3→64, 3×3, s=1)          → (64, 32, 32)
        ┌─ Layer1  3 × Bottleneck(64→64→256)   → (256, 32, 32)  s=1
        ├─ Layer2  4 × Bottleneck(256→128→512)  → (512, 16, 16)  s=2
        ├─ Layer3  6 × Bottleneck(512→256→1024) → (1024,  8,  8)  s=2
        └─ Layer4  3 × Bottleneck(1024→512→2048)→ (2048,  4,  4)  s=2
        GlobalAvgPool → (2048,)
        FC Linear(2048 → num_classes)

    Couches Conv2d accessibles par named_modules() :
      conv1
      layer1.0.conv1/conv2/conv3  + layer1.0.shortcut.0
      layer1.1.conv1/conv2/conv3
      layer1.2.conv1/conv2/conv3
      layer2.0.conv1/conv2/conv3  + layer2.0.shortcut.0
      layer2.1–3.conv1/conv2/conv3
      layer3.0.conv1/conv2/conv3  + layer3.0.shortcut.0
      layer3.1–5.conv1/conv2/conv3
      layer4.0.conv1/conv2/conv3  + layer4.0.shortcut.0
      layer4.1–2.conv1/conv2/conv3

    Total paramètres : ~23.5 M
    (version CIFAR : conv1 3×3 et FC 2048→10 au lieu de 7×7 et 2048→1000)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Première conv adaptée CIFAR (3×3 stride 1 au lieu de 7×7 stride 2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64,    64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(256,  128,  blocks=4, stride=2)
        self.layer3 = self._make_layer(512,  256,  blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512,  blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(2048, num_classes)

    @staticmethod
    def _make_layer(in_ch: int, planes: int,
                    blocks: int, stride: int) -> nn.Sequential:
        layers = [Bottleneck(in_ch, planes, stride=stride)]
        out_ch = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_ch, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
