import torch
import torch.nn as nn

from experiments.nn.resnet3d import ResNet3D
from experiments.nn.resnet_ae import ResNetDecoder
from experiments.nn.utils.meta import run_sequentially


class ResNet3D2DAutoEncoder(nn.Module):

    def __init__(self, decoder_weights='objects365'):
        super(ResNet3D2DAutoEncoder, self).__init__()

        self.encoder3D = ResNet3D()
        self.decoder = ResNetDecoder(weights='objects365')

    def encode(self, x):
        return self.encoder3D(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder3D(x)
        x = self.decoder(x)
        return x
