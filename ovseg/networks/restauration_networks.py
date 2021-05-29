import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# %% helper functions for convolutions
def get_padding(kernel_size):
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        return [(k - 1) // 2 for k in kernel_size]
    else:
        return (kernel_size - 1) // 2


def get_stride(kernel_size):
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        return tuple([(k + 1)//2 for k in kernel_size])
    else:
        return (kernel_size + 1) // 2


# %%
class ResBlock(nn.Module):
    # keeping this class general for the 3d case

    def __init__(self, in_channels, out_channels, is_2d=True, kernel_size=3,
                 first_stride=1, conv_params=None, norm='inst', norm_params=None,
                 nonlin_params=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = get_padding(self.kernel_size)
        self.first_stride = first_stride
        self.is_2d = is_2d
        self.conv_params = conv_params
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params

        if norm is None:
            norm = 'batch' if is_2d else 'inst'

        if self.conv_params is None:
            self.conv_params = {'bias': False}
        if self.nonlin_params is None:
            self.nonlin_params = {'negative_slope': 0.01, 'inplace': True}
        if self.norm_params is None:
            self.norm_params = {'affine': True}
        # init convolutions, normalisation and nonlinearities
        if self.is_2d:
            conv_fctn = nn.Conv2d
            pool_fctn = nn.AvgPool2d
            if norm.lower().startswith('batch'):
                norm_fctn = nn.BatchNorm2d
            elif norm.lower().startswith('inst'):
                norm_fctn = nn.InstanceNorm2d
        else:
            conv_fctn = nn.Conv3d
            pool_fctn = nn.AvgPool3d
            if norm.lower().startswith('batch'):
                norm_fctn = nn.BatchNorm3d
            elif norm.lower().startswith('inst'):
                norm_fctn = nn.InstanceNorm3d
        self.conv1 = conv_fctn(self.in_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               stride=self.first_stride, **self.conv_params)
        self.conv2 = conv_fctn(self.out_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               **self.conv_params)
        self.norm1 = norm_fctn(self.out_channels, **self.norm_params)
        self.norm2 = norm_fctn(self.out_channels, **self.norm_params)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.nonlin1 = nn.LeakyReLU(**self.nonlin_params)
        self.nonlin2 = nn.LeakyReLU(**self.nonlin_params)

        # now the residual connection
        self.downsample = np.prod(self.first_stride) > 1
        if self.downsample and self.in_channels in [1, self.out_channels]:
            # if the number of input channels equal the output channels or if there is only
            # on input channel (first block) we're just average pooling
            self.skip = pool_fctn(self.first_stride, self.first_stride)
        elif self.downsample and self.in_channels != self.out_channels:
            # if the number of channels don't match the skip connection is average pooling 
            # plus 1x1(x1) convolution to match the channel number
            self.skip = nn.Sequential(pool_fctn(self.first_stride, self.first_stride),
                                      conv_fctn(self.in_channels, self.out_channels, 1))
        elif not self.downsample and self.in_channels in [1, self.out_channels]:
            # cool! We don't have to do anything!
            self.skip = nn.Identity()
        else:
            # not downsampling, but channel matching via 1x1(x1) convolutions
            self.skip = conv_fctn(self.in_channels, self.out_channels, 1)

        # now the skip init
        self.a = nn.Parameter(torch.zeros(()))

    def forward(self, xb):
        xb_res = self.nonlin2(self.norm2(self.conv2(self.nonlin1(self.norm1(self.conv1(xb))))))
        return self.skip(xb) + self.a * xb_res

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, kernel_size=3):
        super().__init__()
        
        self.interp = nn.Upsample(scale_factor=get_stride(kernel_size),
                                  mode='bilinear' if is_2d else 'trilinear',
                                  align_corners=True)
        if is_2d:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  padding=get_padding(kernel_size))
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                  padding=get_padding(kernel_size))
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, xb):
        return self.conv(self.interp(xb))


# %%

class ResUNet2d(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, filters=32, filters_max=320, n_stages=5,
                 conv_params=None, norm='inst', norm_params=None, nonlin_params=None,
                 double_channels_every_second_block=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.filters_max = filters_max
        self.n_stages = n_stages
        self.conv_params = conv_params
        self.norm = norm
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.double_channels_every_second_block = double_channels_every_second_block

        if self.double_channels_every_second_block:
            self.filters_list = [min([self.filters * 2 ** (i//2), self.filters_max])
                                 for i in range(self.n_stages)]
        else:
            self.filters_list = [min([self.filters * 2 ** i, self.filters_max])
                                 for i in range(self.n_stages)]

        self.first_strides_list = [1] + (self.n_stages-1)*[2]
        # blocks for the contracting path
        blocks_contr = []
        for in_ch, out_ch, first_stride in zip([self.in_channels] + self.filters_list,
                                               self.filters_list,
                                               self.first_strides_list):
            block = ResBlock(in_ch, out_ch, conv_params=self.conv_params, norm=self.norm,
                             nonlin_params=self.nonlin_params, first_stride=first_stride)
            blocks_contr.append(block)
        self.blocks_contr = nn.ModuleList(blocks_contr)

        # blocks for the expaning path
        blocks_exp = []
        for n_ch in self.filters_list[:-1]:
            block = ResBlock(2 * n_ch, n_ch, conv_params=self.conv_params, norm=self.norm,
                             nonlin_params=self.nonlin_params)
            blocks_exp.append(block)
        self.blocks_exp = nn.ModuleList(blocks_exp)

        # now the upsamplings
        upsamplings = []
        for in_ch, out_ch in zip(self.filters_list[1:], self.filters_list[:-1]):
            up = UpConv(in_ch, out_ch, is_2d=True)
            upsamplings.append(up)
        self.upsamplings = nn.ModuleList(upsamplings)

        self.final_conv = nn.Conv2d(self.filters, self.out_channels, 1, bias=False)

    def forward(self, xb):

        skip_list = []
        for block in self.blocks_contr:
            xb = block(xb)
            skip_list.append(xb)

        for i in range(self.n_stages-2,-1,-1):
            xb = self.upsamplings[i](xb)
            xb = torch.cat([xb, skip_list[i]], 1)
            xb = self.blocks_exp[i](xb)

        return self.final_conv(xb)
