from torch import nn
import torch

from experiments.nn.utils.blocks import DownBlock, UpBlock
from experiments.nn.utils.meta import load_weights, run_sequentially


class TemporalConv2dAE(nn.Module):

    def __init__(self,
                 n_blocks=1,
                 block_size=3,
                 channels=64,
                 action_size=7,
                 weights=None
                 ):
        super(TemporalConv2dAE, self).__init__()

        self.encoder = nn.ModuleList([
            DownBlock(
                n_layers=block_size,
                in_channels=channels + action_size if i == 0 else channels,
                hidden_channels=channels,
                out_channels=channels,
                down_sample=False
            )
            for i in range(n_blocks)
        ])

        self.decoder = nn.ModuleList([
            UpBlock(
                n_layers=block_size,
                in_channels=channels,
                hidden_channels=channels,
                out_channels=channels if i == n_blocks - 1 else channels,
                up_sample=False,
            )
            for i in range(n_blocks)
        ])
        # self.outa = nn.Sigmoid()
        self.outa = nn.ReLU()

        load_weights(self, weights)

    def encode(self, state, action):
        # tiling the action to match state shape
        action_t0 = action[:, :, None, None]
        e_size = state.size()[2]
        action_t0 = torch.tile(action_t0, (1, 1, e_size, e_size))

        # stacking 
        state_action = torch.cat((state, action_t0), dim=1)

        return run_sequentially(self.encoder, state_action)

    def decode(self, encoded_state):
        return self.outa(
            run_sequentially(self.decoder, encoded_state)
        )

    def forward(self, state, action):
        return self.decode(self.encode(state, action))
