import torch
import torch.nn as nn

""" Implementing a basic ResNet residual block with two 3x3 convolutions, batch normalization, and a shortcut path for ResNet. """
class BasicBlock(nn.Module):
    """ Initializing the convolution modules, batch normalization layers, and the downsampling shortcut projection. """
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    """ Running the input through two convolution-BN-ReLU steps and adding the shortcut identity connection. """
    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

""" Implementing a ResNet-style CNN backbone that extracts multi-channel features from input math formula images. """
class MathResNetBackbone(nn.Module):
    """ Initializing the input convolutions, max pooling, and the four sequential stages of residual blocks. """
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=(1, 1))
        self.layer2 = self._make_layer(128, 2, stride=(2, 2))
        self.layer3 = self._make_layer(256, 2, stride=(1, 2)) 
        self.layer4 = self._make_layer(512, 2, stride=(1, 2))

    """ Building a sequence of residual blocks with the specified channel size, stride, and block count. """
    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        """ Looping to instantiate and append subsequent BasicBlocks to the sequential container. """
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    """ Processing the input image sequentially through all conv, pooling, and residual stages. """
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x