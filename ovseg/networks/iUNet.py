import torch.nn as nn


class iUNet(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

    def forward(self, xb):
        raise NotImplementedError('I\'m on strike and I refuse to do any of my work before '
                                  'Christian does his jobs and implements me!!1!!')