import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_padding(kernel_size):
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        return [(k - 1) // 2 for k in kernel_size]
    else:
        return (kernel_size - 1) // 2


class ResBlock(nn.Module):

    def __init__(self, channels, ks1, ks2):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, ks1, padding=get_padding(ks1))
        self.conv2 = nn.Conv3d(channels, channels, ks2, padding=get_padding(ks2))
        
        self.nonlin1 = nn.LeakyReLU(inplace=True)
        self.nonlin2 = nn.LeakyReLU(inplace=True)

        self.a = nn.Parameter(torch.zeros(()))

    def forward(self, xb):
        return xb + self.a * self.conv2(self.nonlin2(self.conv1(self.nonlin1(xb))))


class RefineResNet(nn.Module):

    def __init__(self, in_channels, out_channels, hid_channels, z_to_xy_ratio, n_res_blocks=4,
                 use_large_kernels=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels
        self.z_to_xy_ratio = z_to_xy_ratio
        self.n_res_blocks = n_res_blocks
        self.use_large_kernels = use_large_kernels


        if self.use_large_kernels:
            if self.z_to_xy_ratio < 1.5:
                self.kernel_size_list = [5 for _ in range(2*self.n_res_blocks)]
            else:
                n_3d_convs = int(8 * self.n_res_blocks / self.z_to_xy_ratio+0.5)
                self.kernel_size_list = []
                for i in range(2*self.n_res_blocks):
                    if i < n_3d_convs:
                        self.kernel_size_list.append((3, 5, 5))
                    else:
                        self.kernel_size_list.append((1, 5, 5))
        else:
            if self.z_to_xy_ratio < 1.5:
                self.kernel_size_list = [3 for _ in range(2*self.n_res_blocks)]
            else:
                n_3d_convs = int(4 * self.n_res_blocks / self.z_to_xy_ratio+0.5)
                self.kernel_size_list = []
                for i in range(2*self.n_res_blocks):
                    if i < n_3d_convs:
                        self.kernel_size_list.append((3, 3, 3))
                    else:
                        self.kernel_size_list.append((1, 3, 3))

        self.preprocess = nn.Conv3d(self.in_channels, self.hid_channels, 1)

        res_blocks = []
        for i in range(self.n_res_blocks):
            res_blocks.append(ResBlock(self.hid_channels,
                                       self.kernel_size_list[2*i],
                                       self.kernel_size_list[2*i+1]))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.logits = nn.Conv3d(self.hid_channels, self.out_channels, 1)

    def forward(self, xb):
        return [self.logits(self.res_blocks(self.preprocess(xb)))]
