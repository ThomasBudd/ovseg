import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

GAMMA = np.sqrt(2 / (1 - 1/np.pi))

def get_padding(kernel_size):
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        return [(k - 1) // 2 for k in kernel_size]
    else:
        return (kernel_size - 1) // 2


def get_stride(kernel_size):
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        return [(k + 1)//2 for k in kernel_size]
    else:
        return (kernel_size + 1) // 2


# %%
class nfConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, alpha=0.2, beta=1, kernel_size=3,
                 downsample=False, conv_params=None, nonlin_params=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.kernel_size = kernel_size
        self.padding = get_padding(self.kernel_size)
        self.downsample = downsample
        self.is_2d = is_2d
        self.conv_params = conv_params
        self.nonlin_params = nonlin_params
        if self.downsample:
            self.stride = get_stride(self.kernel_size)
        else:
            self.stride = 1

        if self.conv_params is None:
            self.conv_params = {'bias': True}
        if self.nonlin_params is None:
            self.nonlin_params = {'inplace': True}
        # init convolutions, normalisation and nonlinearities
        if self.is_2d:
            conv_fctn = nn.Conv2d
            pool_fctn = nn.AvgPool2d
        else:
            conv_fctn = nn.Conv3d
            pool_fctn = nn.AvgPool3d

        self.tau = nn.Parameter(torch.zeros(()))
        self.conv1 = conv_fctn(self.in_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               stride=self.stride, **self.conv_params)
        self.conv2 = conv_fctn(self.out_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               **self.conv_params)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.nonlin1 = nn.ReLU(**self.nonlin_params)
        self.nonlin2 = nn.ReLU(**self.nonlin_params)

        if self.downsample and self.in_channels == self.out_channels:
            self.skip = pool_fctn(self.stride, self.stride)
        elif self.downsample and self.in_channels != self.out_channels:
            self.skip = nn.Sequential(pool_fctn(self.stride, self.stride),
                                      conv_fctn(self.in_channels, self.out_channels, 1))
        elif not self.downsample and self.in_channels == self.out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = conv_fctn(self.in_channels, self.out_channels, 1)

    def forward(self, xb):
        skip = self.skip(xb)
        xb = self.nonlin1(xb / self.beta) * GAMMA
        xb = self.conv1(xb)
        xb = self.nonlin2(xb) * GAMMA
        xb = self.conv2(xb) * self.alpha * self.tau
        return xb + skip


class nfConvStage(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, n_blocks=1,
                 alpha=0.2, beta=1, kernel_size=3,
                 downsample=False, conv_params=None, nonlin_params=None):
        super().__init__()
        downsample_list = [downsample] + [False for _ in range(n_blocks-1)]
        in_channels_list = [in_channels] + [out_channels for _ in range(n_blocks-1)]
        blocks_list = []
        for downsample, in_channels in zip(downsample_list, in_channels_list):
            blocks_list.append(nfConvBlock(in_channels, out_channels, is_2d,
                                           alpha=alpha, beta=beta, kernel_size=kernel_size,
                                           downsample=downsample, conv_params=conv_params,
                                           nonlin_params=nonlin_params))
        self.blocks = nn.ModuleList(blocks_list)

    def forward(self, xb):
        for block in self.blocks:
            xb = block(xb)
        return xb


# %% transposed convolutions
class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, kernel_size=2):
        super().__init__()
        if is_2d:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size, stride=kernel_size,
                                           bias=False)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                                           kernel_size, stride=kernel_size,
                                           bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, xb):
        return self.conv(xb)


# %% now simply the logits
class Logits(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d):
        super().__init__()
        if is_2d:
            self.logits = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.logits = nn.Conv3d(in_channels, out_channels, 1)
        nn.init.kaiming_normal_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, xb):
        return self.logits(xb)


# %%
class nfUNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes,
                 is_2d, n_blocks=None, filters=32, filters_max=384, n_pyramid_scales=None,
                 conv_params=None, nonlin_params=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.is_2d = is_2d
        self.n_stages = len(kernel_sizes)
        if n_blocks is None:
            self.n_blocks = [1 for _ in range(self.n_stages)]
        else:
            assert len(n_blocks) == self.n_stages
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.nonlin_params = nonlin_params
        # we double the amount of channels every downsampling step
        # up to a max of filters_max
        self.filters_list = [min([self.filters*2**i, self.filters_max])
                             for i in range(self.n_stages)]
        # determine how many scales on the upwars path with be connected to
        # a loss function
        if n_pyramid_scales is None:
            if self.n_stages > 2:
                self.n_pyramid_scales = self.n_stages - 2
            else:
                self.n_pyramid_scales = 1
        else:
            self.n_pyramid_scales = int(n_pyramid_scales)

        # first only the two blocks at the top
        self.blocks_down = []
        block = nfConvStage(self.in_channels, self.filters, self.is_2d,
                            n_blocks=self.n_blocks[0],
                            kernel_size=self.kernel_sizes[0],
                            downsample=False,
                            conv_params=self.conv_params,
                            nonlin_params=self.nonlin_params)
        self.blocks_down.append(block)
        self.blocks_up = []
        block = nfConvStage(2*self.filters, self.filters, self.is_2d,
                            n_blocks=1,
                            kernel_size=self.kernel_sizes[0],
                            downsample=False,
                            conv_params=self.conv_params,
                            nonlin_params=self.nonlin_params)
        self.blocks_up.append(block)

        self.upconvs = []
        self.all_logits = []

        # now all the others incl upsampling and logits
        for i, ks in enumerate(self.kernel_sizes[1:]):

            # down block
            block = nfConvStage(self.filters_list[i],
                                self.filters_list[i+1],
                                self.is_2d,
                                n_blocks=self.n_blocks[i+1],
                                kernel_size=ks,
                                downsample=True,
                                conv_params=self.conv_params,
                                nonlin_params=self.nonlin_params)
            self.blocks_down.append(block)

            # block on the upwards pass except for the bottom stage
            if i < self.n_stages - 1:
                block = nfConvStage(2 * self.filters_list[i+1],
                                    self.filters_list[i+1],
                                    self.is_2d,
                                    n_blocks=1,
                                    kernel_size=ks,
                                    downsample=False,
                                    conv_params=self.conv_params,
                                    nonlin_params=self.nonlin_params)
                self.blocks_up.append(block)
            # convolutions to this stage
            upconv = UpConv(self.filters_list[i+1], self.filters_list[i],
                            is_2d, get_stride(ks))
            self.upconvs.append(upconv)
            # logits for this stage
            if i < self.n_pyramid_scales:
                logits = Logits(self.filters_list[i], self.out_channels, is_2d)
                self.all_logits.append(logits)

        # now important let's turn everything into a module list
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upconvs = nn.ModuleList(self.upconvs)
        self.all_logits = nn.ModuleList(self.all_logits)

    def forward(self, xb):
        # keep all out tensors from the contracting path
        xb_list = []
        logs_list = []
        # contracting path
        for i in range(self.n_stages):
            xb = self.blocks_down[i](xb)
            xb_list.append(xb)

        # expanding path without logits
        for i in range(self.n_stages - 2, self.n_pyramid_scales-1, -1):
            xb = self.upconvs[i](xb)
            xb = torch.cat([xb, xb_list[i]], 1)
            xb = self.blocks_up[i](xb)

        # expanding path with logits
        for i in range(self.n_pyramid_scales - 1, -1, -1):
            xb = self.upconvs[i](xb)
            xb = torch.cat([xb, xb_list[i]], 1)
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i](xb)
            logs_list.append(logs)

        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]
