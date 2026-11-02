import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Bloc résiduel 2 couches conv (ResNet-18/34)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet18Cifar(nn.Module):
    """ResNet-18 adapté pour CIFAR-10 (entrée 32×32).

    Différences vs ResNet-18 ImageNet :
      - conv1 : kernel 3×3, stride 1, padding 1  (au lieu de 7×7/2)
      - pas de MaxPool2d après conv1
      - GlobalAvgPool2d en fin → compatible 32×32

    Couches Conv2d accessibles par named_modules() :
      conv1
      layer1.0.conv1 / conv2
      layer1.1.conv1 / conv2
      layer2.0.conv1 / conv2  + layer2.0.shortcut.0
      layer2.1.conv1 / conv2
      layer3.0.conv1 / conv2  + layer3.0.shortcut.0
      layer3.1.conv1 / conv2
      layer4.0.conv1 / conv2  + layer4.0.shortcut.0
      layer4.1.conv1 / conv2
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Première conv adaptée CIFAR (3×3 stride 1 au lieu de 7×7 stride 2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64,  64,  stride=1)
        self.layer2 = self._make_layer(64,  128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            BasicBlock(in_ch, out_ch, stride=stride),
            BasicBlock(out_ch, out_ch, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
