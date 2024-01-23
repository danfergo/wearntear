from collections import namedtuple

import torch
from torch import nn


class AssociativeCatAE(nn.Module):
    """
        Associates Vision and Touch with plane concatenation
    """

    def __init__(self):
        super(AssociativeCatAE, self).__init__()

    def associate(self, vision, touch):
        return torch.cat((vision, touch), dim=1)

    def dissociate(self, encoded_state):
        m = encoded_state.shape[1] // 2  # middle point, in the channels dimension
        return encoded_state[:, :m, :, :], encoded_state[:, m:, :, :]

    def forward(self, vision, touch):
        return self.dissociate(self.associate(vision, touch))
