import torch
import torch.nn as nn

from experiments.nn.utils.meta import run_sequentially


class ResNet3D(nn.Module):

    def __init__(self, n_frames=3):
        super(ResNet3D, self).__init__()

        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        # model.blocks[5].proj = nn.Linear(in_features=2048, out_features=n_activations, bias=True)

        self.conv2d = torch.nn.Conv2d(n_frames * 1024, 2048, kernel_size=3, padding='same')

        torch.nn.init.xavier_uniform_(self.conv2d.weight)
        self.m = nn.MaxPool3d((1, 2, 2))

        self.conv2d.bias.data.fill_(0.0)
        self.bn = nn.BatchNorm2d(2048)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = run_sequentially(self.model.blocks[0:4], x)

        x = self.m(x)
        s = x.size()
        x = x.view(s[0], s[1] * s[2], s[3], s[4])
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
