import sys
import os
import torch


# def force_cudnn_initialization():
#     s = 32
#     dev = torch.device('cuda')
#     torch.nn.functional.conv(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
#     print('[dummy cuda init]')
