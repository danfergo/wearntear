from torch import nn
import torch

from experiments.nn.utils.blocks import DownBlock, UpBlock
from experiments.nn.utils.meta import load_weights, run_sequentially


class Convert3Dto2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Convert3Dto2D, self).__init__()

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')

        self.conv2d.bias.data.fill_(0.0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        s = x.size()
        x = x.view(s[0], s[1] * s[2], s[3], s[4])
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SensoryAE(nn.Module):

    def __init__(self,
                 channels=64,
                 n_blocks=3,
                 block_size=3,
                 weights=None,
                 stack_size=4,
                 img_channels=3,
                 encoder_shape='3d'):
        super(SensoryAE, self).__init__()

        self.encoder = nn.ModuleList([
            DownBlock(
                n_layers=block_size,
                in_channels=(stack_size if encoder_shape == '3d' else img_channels) if i == 0 else channels,
                hidden_channels=channels,
                out_channels=channels,
                shape=encoder_shape,
                down_sample=False
            )
            for i in range(n_blocks)
        ])
        self.convert_es_3dto2d = Convert3Dto2D(img_channels * channels, channels)
        self.convert_skip_3dto2d = Convert3Dto2D(img_channels * channels, channels)

        # self.fourier_features = None
        # if use_fourier_features:
        #     self.fourier_features = FourierFeatures2D(channels, channels)
        #     channels = 2 * channels

        self.decoder = nn.ModuleList([
            UpBlock(
                n_layers=block_size,
                in_channels=channels,
                hidden_channels=channels,
                out_channels=channels,
                up_sample=False
                # up_sample=i < n_blocks,
                # use_last_activation=i < n_blocks
            )
            for i in range(n_blocks)
        ])

        self.outc = nn.Conv2d(channels, img_channels, kernel_size=3, padding='same')
        self.outa = nn.Sigmoid()
        # self.outa = nn.ReLU()

        load_weights(self, weights)

    def encode(self, x, return_hidden=False):
        e3d, h = run_sequentially(self.encoder, x, return_hidden)
        e2d = self.convert_es_3dto2d(e3d)
        return e2d, h

    def decode(self, encoded_vision, hidden_values=None):
        def _fuse_hidden(ht, i):
            if i == len(self.decoder) - 1:
                skip2d = self.convert_skip_3dto2d(hidden_values[0])
                return skip2d + ht
            return ht

        return self.outa(
            self.outc(
                run_sequentially(
                    self.decoder,
                    encoded_vision,
                    map_fn=_fuse_hidden if hidden_values else None
                )
            )
        )

    def forward(self, x, skip_connections=False):
        return self.decode(self.encode(x, skip_connections))
