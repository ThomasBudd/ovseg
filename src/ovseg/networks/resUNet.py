import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ovseg.networks.blocks import StochDepth, SE_unit
from ovseg.networks.custom_normalization import no_z_InstNorm, my_LayerNorm
import os
import pickle

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
class ConvNormNonlinBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, kernel_size=3,
                 first_stride=1, conv_params=None, norm=None, norm_params=None,
                 nonlin_params=None, hid_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels if hid_channels is not None else out_channels
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
            if norm.lower().startswith('batch'):
                norm_fctn = nn.BatchNorm2d
            elif norm.lower().startswith('inst'):
                norm_fctn = nn.InstanceNorm2d
        else:
            conv_fctn = nn.Conv3d
            if norm.lower().startswith('batch'):
                norm_fctn = nn.BatchNorm3d
            elif norm.lower().startswith('inst'):
                norm_fctn = nn.InstanceNorm3d
            elif norm.lower().startswith('no_z_'):
                norm_fctn = no_z_InstNorm
            elif norm.lower().startswith('layer'):
                norm_fctn = my_LayerNorm
        self.conv1 = conv_fctn(self.in_channels, self.hid_channels,
                               self.kernel_size, padding=self.padding,
                               stride=self.first_stride, **self.conv_params)
        self.conv2 = conv_fctn(self.hid_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               **self.conv_params)
        self.norm1 = norm_fctn(self.hid_channels, **self.norm_params)
        self.norm2 = norm_fctn(self.out_channels, **self.norm_params)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.nonlin1 = nn.LeakyReLU(**self.nonlin_params)
        self.nonlin2 = nn.LeakyReLU(**self.nonlin_params)

    def forward(self, xb):
        xb = self.conv1(xb)
        xb = self.norm1(xb)
        xb = self.nonlin1(xb)
        xb = self.conv2(xb)
        xb = self.norm2(xb)
        xb = self.nonlin2(xb)
        return xb

class ConvNormNonlinBlockV2(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, kernel_size=3,
                 first_stride=1, conv_params=None, norm=None, norm_params=None,
                 nonlin_params=None, hid_channels=None):
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
            if norm.lower().startswith('batch'):
                norm_fctn = nn.BatchNorm2d
            elif norm.lower().startswith('inst'):
                norm_fctn = nn.InstanceNorm2d
        else:
            conv_fctn = nn.Conv3d
            if norm.lower().startswith('batch'):
                norm_fctn = nn.BatchNorm3d
            elif norm.lower().startswith('inst'):
                norm_fctn = nn.InstanceNorm3d
            elif norm.lower().startswith('no_z_'):
                norm_fctn = no_z_InstNorm
            elif norm.lower().startswith('layer'):
                norm_fctn = my_LayerNorm
        self.conv = conv_fctn(self.in_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               stride=self.first_stride, **self.conv_params)
        self.norm = norm_fctn(self.out_channels, **self.norm_params)

        nn.init.kaiming_normal_(self.conv.weight)
        self.nonlin = nn.LeakyReLU(**self.nonlin_params)

    def forward(self, xb):
        xb = self.conv(xb)
        xb = self.norm(xb)
        xb = self.nonlin(xb)
        return xb

# %%
class ResBlock(nn.Module):
    # keeping this class general for the 3d case

    def __init__(self, in_channels, out_channels, is_2d=False, kernel_size=3, kernel_size2=None,
                 first_stride=1, conv_params=None, norm='inst', norm_params=None,
                 nonlin_params=None, stochdepth_rate=0.2, use_se=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size2 = kernel_size if kernel_size2 is None else kernel_size2
        self.first_stride = first_stride
        self.is_2d = is_2d
        self.conv_params = conv_params
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.stochdepth_rate = stochdepth_rate
        self.use_se = use_se

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
            elif norm.lower().startswith('no_z_'):
                norm_fctn = no_z_InstNorm
            elif norm.lower().startswith('layer'):
                norm_fctn = my_LayerNorm
        self.conv1 = conv_fctn(self.in_channels, self.out_channels,
                               self.kernel_size, padding=get_padding(self.kernel_size),
                               stride=self.first_stride, **self.conv_params)
        self.conv2 = conv_fctn(self.out_channels, self.out_channels,
                               self.kernel_size2, padding=get_padding(self.kernel_size2),
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
        self.stochdepth = StochDepth(self.stochdepth_rate)
        if self.use_se:
            self.se = SE_unit(self.out_channels, is_2d=self.is_2d)

    def forward(self, xb):
        xb_res = self.nonlin2(self.norm2(self.conv2(self.nonlin1(self.norm1(self.conv1(xb))))))
        if self.use_se:
            xb_res = self.se(xb_res)
        return self.skip(xb) + self.a * self.stochdepth(xb_res)


class ResBottleneckBlock(nn.Module):
    # keeping this class general for the 3d case

    def __init__(self, in_channels, out_channels, is_2d=False, kernel_size=3, kernel_size2=None,
                 first_stride=1, conv_params=None, norm='inst', norm_params=None,
                 nonlin_params=None, bottleneck_ratio=2, stochdepth_rate=0.2, use_se=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size2 = kernel_size if kernel_size2 is None else kernel_size2
        self.first_stride = first_stride
        self.is_2d = is_2d
        self.conv_params = conv_params
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.bottleneck_ratio = bottleneck_ratio
        self.hid_channels = self.out_channels // self.bottleneck_ratio
        self.stochdepth_rate = stochdepth_rate
        self.use_se = use_se

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
            elif norm.lower().startswith('no_z_'):
                norm_fctn = no_z_InstNorm
            elif norm.lower().startswith('layer'):
                norm_fctn = my_LayerNorm
        self.conv1 = conv_fctn(self.in_channels, self.hid_channels, 1, **self.conv_params)
        self.conv2 = conv_fctn(self.hid_channels, self.hid_channels,
                               self.kernel_size, padding=get_padding(self.kernel_size),
                               stride=self.first_stride, **self.conv_params)
        self.conv3 = conv_fctn(self.hid_channels, self.hid_channels,
                               self.kernel_size2, padding=get_padding(self.kernel_size2),
                               **self.conv_params)
        self.conv4 = conv_fctn(self.hid_channels, self.out_channels, 1, **self.conv_params)
        self.norm1 = norm_fctn(self.hid_channels, **self.norm_params)
        self.norm2 = norm_fctn(self.hid_channels, **self.norm_params)
        self.norm3 = norm_fctn(self.hid_channels, **self.norm_params)
        self.norm4 = norm_fctn(self.out_channels, **self.norm_params)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        self.nonlin1 = nn.LeakyReLU(**self.nonlin_params)
        self.nonlin2 = nn.LeakyReLU(**self.nonlin_params)
        self.nonlin3 = nn.LeakyReLU(**self.nonlin_params)
        self.nonlin4 = nn.LeakyReLU(**self.nonlin_params)

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
        self.stochdepth = StochDepth(self.stochdepth_rate)
        if self.use_se:
            self.se = SE_unit(self.out_channels, is_2d=self.is_2d)

    def forward(self, xb):

        skip = self.skip(xb)
        xb = self.nonlin1(self.norm1(self.conv1(xb)))
        xb = self.nonlin2(self.norm2(self.conv2(xb)))
        xb = self.nonlin3(self.norm3(self.conv3(xb)))
        xb = self.nonlin4(self.norm4(self.conv4(xb)))
        if self.use_se:
            xb = self.se(xb)


        return skip + self.a * self.stochdepth(xb)

# %%
class MergeAndRunBlock(nn.Module):
    # keeping this class general for the 3d case

    def __init__(self, in_channels, out_channels, is_2d=False, kernel_size=3, kernel_size2=None,
                 first_stride=1, conv_params=None, norm='inst', norm_params=None,
                 nonlin_params=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size2 = kernel_size if kernel_size2 is None else kernel_size2
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
            elif norm.lower().startswith('no_z_'):
                norm_fctn = no_z_InstNorm
            elif norm.lower().startswith('layer'):
                norm_fctn = my_LayerNorm
        self.conv1 = conv_fctn(self.in_channels, self.out_channels, self.kernel_size,
                               padding=get_padding(self.kernel_size), stride=self.first_stride,
                               groups=2, **self.conv_params)
        self.conv2 = conv_fctn(self.out_channels, self.out_channels, self.kernel_size2,
                               padding=get_padding(self.kernel_size2), groups=2,
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

    def forward(self, xb): 
        skip = self.skip(xb)
        skip = 0.5 * (skip + torch.cat([skip[:, self.out_channels//2:], 
                                        skip[:, :self.out_channels//2]], 1))
        xb_res = self.nonlin2(self.norm2(self.conv2(self.nonlin1(self.norm1(self.conv1(xb))))))
        
        return skip + xb_res

# %%
class stackedResBlocks(nn.Module):

    def __init__(self, block, n_blocks, in_channels, out_channels, init_stride, is_2d=False,
                 z_to_xy_ratio=1, conv_params=None, norm='inst', norm_params=None, 
                 nonlin_params=None, bottleneck_ratio=2, stochdepth_rate=0.0, use_se=False):
        super().__init__()
        assert block in ['res', 'bottleneck', 'mergeandrun']
        self.block = block
        self.n_blocks = n_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_stride = init_stride
        self.is_2d = is_2d
        if not self.is_2d:
            self.z_to_xy_ratio = max([int(z_to_xy_ratio+0.5), 1])
        self.conv_params = conv_params
        self.norm = norm
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.use_se = use_se
        if self.block == 'bottleneck':
            self.bottleneck_ratio = bottleneck_ratio
        if not self.block == 'mergeandrun':
            self.stochdepth_rate = stochdepth_rate

        # now create the lists we need for creating the blocks
        in_channels_list = [self.in_channels] + (self.n_blocks - 1) * [self.out_channels]
        init_stride_list = [self.init_stride] + (self.n_blocks -1) * [1]

        kernel_sizes_list = (2 * self.n_blocks) * [3]
        if not self.is_2d and self.z_to_xy_ratio > 1:
            # in this case we want to apply some inplane convolutions
            for i in range(1, 2*self.n_blocks + 1):
                if i % self.z_to_xy_ratio != 0:
                    kernel_sizes_list[i-1] = (1, 3, 3)

        res_blocks = []
        if self.block == 'res':
            res_block = ResBlock
            kwargs = {'stochdepth_rate': self.stochdepth_rate,
                      'use_se': self.use_se}
        elif self.block == 'bottleneck':
            res_block = ResBottleneckBlock
            kwargs = {'stochdepth_rate': self.stochdepth_rate,
                      'bottleneck_ratio': self.bottleneck_ratio,
                      'use_se': self.use_se}
        elif self.block == 'mergeandrun':
            res_block = MergeAndRunBlock
            kwargs = {}
        for i in range(self.n_blocks):
            res_blocks.append(res_block(in_channels=in_channels_list[i],
                                        out_channels=self.out_channels,
                                        is_2d=self.is_2d,
                                        kernel_size=kernel_sizes_list[2*i],
                                        kernel_size2=kernel_sizes_list[2*i+1],
                                        first_stride=init_stride_list[i],
                                        conv_params=self.conv_params,
                                        norm=self.norm,
                                        norm_params=self.norm_params,
                                        nonlin_params=self.nonlin_params,
                                        **kwargs))
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, xb):
        return self.res_blocks(xb)

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

class UpConv33(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=(1, 3, 3),
                                        stride=(1, 2, 2),
                                        bias=False)
        self.conv2 = nn.ConvTranspose3d(in_channels, out_channels,
                                        kernel_size=1,
                                        stride=(1, 2, 2),
                                        bias=False)
        nn.init.kaiming_normal_(self.conv.weight1)
        nn.init.kaiming_normal_(self.conv.weight2)

    def forward(self, xb):
        return self.conv1(xb) + self.conv2(xb)

class UpLinear(nn.Module):

    def __init__(self, kernel_size, is_2d):
        
        if is_2d:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='trilinear',
                                  align_corners=True)
    def forward(self, xb):
        return self.up(xb)


# %%
class PixelShuffle3d(nn.Module):
    
    def __init__(self, r=2):
        super().__init__()
        self.r = r
        self.shuffle = nn.PixelShuffle(self.r)

    def forward(self, xb):
        
        return torch.stack([self.shuffle(xb[:, :, z]) for z in range(xb.shape[2])], 2)

class PixelUnshuffle3d(nn.Module):
    
    def __init__(self, r=2):
        super().__init__()
        self.r = r
        self.shuffle = nn.PixelUnshuffle(self.r)

    def forward(self, xb):
        
        return torch.stack([self.shuffle(xb[:, :, z]) for z in range(xb.shape[2])], 2)


# %% now simply the logits
class Logits(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, p_dropout=0, use_bias=False):
        super().__init__()
        if is_2d:
            self.logits = nn.Conv2d(in_channels, out_channels, 1, bias=use_bias)
            self.dropout = nn.Dropout2d(p_dropout, inplace=True)
        else:
            self.logits = nn.Conv3d(in_channels, out_channels, 1, bias=use_bias)
            self.dropout = nn.Dropout3d(p_dropout, inplace=True)
        nn.init.kaiming_normal_(self.logits.weight)

    def forward(self, xb):
        return self.dropout(self.logits(xb))


# %%
class UNetResDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, z_to_xy_ratio, block='res',
                 n_blocks_list=[1, 2, 6, 3], filters=32, filters_max=384,
                 conv_params=None, norm=None, norm_params=None, nonlin_params=None, 
                 bottleneck_ratio=2, stochdepth_rate=0.0, p_dropout_logits=0.0, use_se=False,
                 use_logit_bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_2d = is_2d
        self.block = block
        self.z_to_xy_ratio = z_to_xy_ratio
        self.n_blocks_list = n_blocks_list
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.norm = norm
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.p_dropout_logits = p_dropout_logits
        self.bottleneck_ratio = bottleneck_ratio
        self.stochdepth_rate = stochdepth_rate
        self.use_se = use_se
        self.use_logit_bias = use_logit_bias
        # we double the amount of channels every downsampling step
        # up to a max of filters_max
        self.n_stages = len(n_blocks_list)
        self.filters_list = [min([self.filters*2**i, self.filters_max])
                             for i in range(self.n_stages)]

        # first let's make the lists for the blocks on both pathes
        # number of input and output channels of the blocks on the contracting path
        self.in_channels_down_list = [self.in_channels] + self.filters_list[:-1]
        self.in_channels_up_list = [2*f for f in self.filters_list[:-1]]
        self.out_channels_list = self.filters_list
        self.z_to_xy_ratio_list = [max([self.z_to_xy_ratio / 2**i, 1])
                                   for i in range(self.n_stages)]
        if self.is_2d:
            self.kernel_sizes_up = (self.n_stages-1) * [3]
        else:
            self.kernel_sizes_up = [(1, 3, 3) if z_to_xy >= 2 else 3 for z_to_xy in 
                                    self.z_to_xy_ratio_list]
        self.init_stride_list = [1] + [get_stride(ks) for ks in self.kernel_sizes_up]


        # blocks on the contracting path
        self.blocks_down = []
        for in_ch, out_ch, init_stride, z_to_xy in zip(self.in_channels_down_list[:-1],
                                                       self.out_channels_list[:-1],
                                                       self.init_stride_list[:-1],
                                                       self.z_to_xy_ratio_list[:-1]):
            kernel_size = (1, 3, 3) if not self.is_2d and z_to_xy >= 2 else 3
            self.blocks_down.append(ConvNormNonlinBlock(in_channels=in_ch,
                                                        out_channels=out_ch,
                                                        first_stride=init_stride,
                                                        kernel_size=kernel_size,
                                                        is_2d=self.is_2d,
                                                        conv_params=self.conv_params,
                                                        norm=self.norm,
                                                        norm_params=self.norm_params,
                                                        nonlin_params=self.nonlin_params))
        # the lowest block will be residual style
        self.blocks_down.append(stackedResBlocks(block=self.block,
                                                 n_blocks=self.n_blocks_list[-1],
                                                 in_channels=self.in_channels_down_list[-1],
                                                 out_channels=self.out_channels_list[-1],
                                                 init_stride=self.init_stride_list[-1],
                                                 is_2d=self.is_2d,
                                                 z_to_xy_ratio=self.z_to_xy_ratio_list[-1],
                                                 conv_params=self.conv_params,
                                                 norm=self.norm,
                                                 norm_params=self.norm_params,
                                                 nonlin_params=self.nonlin_params,
                                                 bottleneck_ratio=self.bottleneck_ratio,
                                                 stochdepth_rate=self.stochdepth_rate,
                                                 use_se=self.use_se))


        # blocks on the upsampling path
        self.blocks_up = []
        # uppest block on the upsampling path we're doing normal again
        kernel_size = (1, 3, 3) if not self.is_2d and self.z_to_xy_ratio >= 2 else 3
        block = ConvNormNonlinBlock(in_channels=self.in_channels_up_list[0],
                                    out_channels=self.out_channels_list[0],
                                    is_2d=self.is_2d,
                                    kernel_size=kernel_size,
                                    conv_params=self.conv_params,
                                    norm=self.norm,
                                    norm_params=self.norm_params,
                                    nonlin_params=self.nonlin_params)
        self.blocks_up.append(block)
        # the other will be residual
        for in_channels, out_channels, n_blocks, z_to_xy in zip(self.in_channels_up_list[1:],
                                                                self.out_channels_list[1:-1],
                                                                self.n_blocks_list[1:-1],
                                                                self.z_to_xy_ratio_list[1:-1]):
            block = stackedResBlocks(block=self.block,
                                     n_blocks=n_blocks,
                                     in_channels=in_channels,
                                     out_channels=out_channels,
                                     is_2d=self.is_2d,
                                     init_stride=1,
                                     z_to_xy_ratio=z_to_xy,
                                     conv_params=self.conv_params,
                                     norm=self.norm,
                                     norm_params=self.norm_params,
                                     nonlin_params=self.nonlin_params,
                                     bottleneck_ratio=self.bottleneck_ratio,
                                     stochdepth_rate=self.stochdepth_rate,
                                     use_se=self.use_se)
            self.blocks_up.append(block)

        # upsaplings
        self.upsamplings = []
        for in_channels, out_channels, kernel_size in zip(self.out_channels_list[1:],
                                                          self.out_channels_list[:-1],
                                                          self.kernel_sizes_up):
            self.upsamplings.append(UpConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           is_2d=self.is_2d,
                                           kernel_size=get_stride(kernel_size)))
        # logits
        self.all_logits = []
        for in_channels in self.out_channels_list:
            self.all_logits.append(Logits(in_channels=in_channels,
                                          out_channels=self.out_channels,
                                          is_2d=self.is_2d,
                                          p_dropout=self.p_dropout_logits,
                                          use_bias=self.use_logit_bias))

        # now important let's turn everything into a module list
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upsamplings = nn.ModuleList(self.upsamplings)
        self.all_logits = nn.ModuleList(self.all_logits)

    def forward(self, xb):
        # keep all out tensors from the contracting path
        xb_list = []
        logs_list = []
        # contracting path
        for block in self.blocks_down[:-1]:
            xb = block(xb)
            xb_list.append(xb)

        # bottom block
        xb = self.blocks_down[-1](xb)
        logs_list.append(self.all_logits[-1](xb))

        # expanding path with logits
        for i in range(self.n_stages - 2, -1, -1):
            xb = self.upsamplings[i](xb)
            xb = torch.cat([xb, xb_list[i]], 1)
            del xb_list[i]
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i](xb)
            logs_list.append(logs)

        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]

    def update_prg_trn(self, param_dict, h, indx=None):
        if 'p_dropout_logits' in param_dict:
            p = (1 - h) * param_dict['p_dropout_logits'][0] + h * param_dict['p_dropout_logits'][1]
            for l in self.all_logits:
                l.dropout.p = p

# %%
class UNetResStemEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, z_to_xy_ratio, block='res',
                 n_blocks_list=[1, 1, 2, 6, 3], filters=16, filters_max=384,
                 conv_params=None, norm=None, norm_params=None, nonlin_params=None, 
                 bottleneck_ratio=2, stochdepth_rate=0.0, p_dropout_logits=0.0, use_se=False,
                 use_logit_bias=False, final_upsampling_33=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_2d = is_2d
        self.block = block
        self.z_to_xy_ratio = z_to_xy_ratio
        self.n_blocks_list = n_blocks_list
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.norm = norm
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.p_dropout_logits = p_dropout_logits
        self.bottleneck_ratio = bottleneck_ratio
        self.stochdepth_rate = stochdepth_rate
        self.use_se = use_se
        self.use_logit_bias = use_logit_bias
        self.final_upsampling_33 = final_upsampling_33
        # we double the amount of channels every downsampling step
        # up to a max of filters_max
        self.n_stages = len(n_blocks_list)
        assert self.n_blocks_list[0] == 1, 'The stem version requires only one block on top'
        self.filters_list = [min([self.filters*2**i, self.filters_max])
                             for i in range(self.n_stages)]

        # first let's make the lists for the blocks on both pathes
        # number of input and output channels of the blocks on the contracting path
        self.in_channels_down_list = [self.in_channels] + self.filters_list[:-1]
        self.in_channels_up_list = [2*f for f in self.filters_list[:-1]]
        self.out_channels_list = self.filters_list
        self.z_to_xy_ratio_list = [max([self.z_to_xy_ratio / 2**i, 1])
                                   for i in range(self.n_stages)]
        if self.is_2d:
            self.kernel_sizes_up = (self.n_stages-1) * [3]
        else:
            self.kernel_sizes_up = [(1, 3, 3) if z_to_xy >= 2 else 3 for z_to_xy in 
                                    self.z_to_xy_ratio_list]
        self.init_stride_list = [1, 1] + [get_stride(ks) for ks in self.kernel_sizes_up[1:]]


        # blocks on the contracting path
        
        self.first_conv = nn.Conv3d(self.in_channels, 
                                    self.filters,
                                    kernel_size=self.kernel_sizes_up[0],
                                    stride=get_stride(self.kernel_sizes_up[0]),
                                    padding=get_padding(self.kernel_sizes_up[0]))
        
        self.blocks_down = []
        for n_blocks, in_ch, out_ch, init_stride, z_to_xy in zip(self.n_blocks_list[1:],
                                                                 self.in_channels_down_list[1:],
                                                                 self.out_channels_list[1:],
                                                                 self.init_stride_list[1:],
                                                                 self.z_to_xy_ratio_list[1:]):
            self.blocks_down.append(stackedResBlocks(block=self.block,
                                                     n_blocks=n_blocks,
                                                     in_channels=in_ch,
                                                     out_channels=out_ch,
                                                     init_stride=init_stride,
                                                     is_2d=self.is_2d,
                                                     z_to_xy_ratio=z_to_xy,
                                                     conv_params=self.conv_params,
                                                     norm=self.norm,
                                                     norm_params=self.norm_params,
                                                     nonlin_params=self.nonlin_params,
                                                     bottleneck_ratio=self.bottleneck_ratio,
                                                     stochdepth_rate=self.stochdepth_rate,
                                                     use_se=self.use_se))

        # blocks on the upsampling path
        self.blocks_up = []
        for in_channels, out_channels, kernel_size in zip(self.in_channels_up_list[1:],
                                                          self.out_channels_list[1:],
                                                          self.kernel_sizes_up[1:]):
            block = ConvNormNonlinBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        is_2d=self.is_2d,
                                        kernel_size=kernel_size,
                                        conv_params=self.conv_params,
                                        norm=self.norm,
                                        norm_params=self.norm_params,
                                        nonlin_params=self.nonlin_params)
            self.blocks_up.append(block)
        
        if self.final_upsampling_33:
            self.last_conv = UpConv33(2*self.filters, self.out_channels)
        else:
            self.last_conv = UpConv(2*self.filters, self.out_channels, self.is_2d,
                                    get_stride(self.kernel_sizes_up[0]))
            

        # upsaplings
        self.upsamplings = []
        for in_channels, out_channels, kernel_size in zip(self.out_channels_list[2:],
                                                          self.out_channels_list[1:-1],
                                                          self.kernel_sizes_up[1:]):
            self.upsamplings.append(UpConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           is_2d=self.is_2d,
                                           kernel_size=get_stride(kernel_size)))
        # logits
        self.all_logits = []
        for in_channels in self.out_channels_list[1:]:
            self.all_logits.append(Logits(in_channels=in_channels,
                                          out_channels=self.out_channels,
                                          is_2d=self.is_2d,
                                          p_dropout=self.p_dropout_logits,
                                          use_bias=self.use_logit_bias))

        # now important let's turn everything into a module list
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upsamplings = nn.ModuleList(self.upsamplings)
        self.all_logits = nn.ModuleList(self.all_logits)

    def forward(self, xb):
        # keep all out tensors from the contracting path
        
        xb = self.first_conv(xb)
        
        xb_list = []
        logs_list = []
        # contracting path
        for block in self.blocks_down[:-1]:
            xb = block(xb)
            xb_list.append(xb)

        # bottom block
        xb = self.blocks_down[-1](xb)
        logs_list.append(self.all_logits[-1](xb))

        # expanding path with logits
        for i in range(self.n_stages - 3, -1, -1):
            xb = self.upsamplings[i](xb)
            xb = torch.cat([xb, xb_list[i]], 1)
            del xb_list[i]
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i](xb)
            logs_list.append(logs)

        logs_list.append(self.last_conv(xb))        
        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]

    def update_prg_trn(self, param_dict, h, indx=None):
        if 'p_dropout_logits' in param_dict:
            p = (1 - h) * param_dict['p_dropout_logits'][0] + h * param_dict['p_dropout_logits'][1]
            for l in self.all_logits:
                l.dropout.p = p


# %%
class UNetResShuffleEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, z_to_xy_ratio, block='res',
                 n_blocks_list=[1, 1, 2, 6, 3], filters=16, filters_max=384,
                 conv_params=None, norm=None, norm_params=None, nonlin_params=None, 
                 bottleneck_ratio=2, stochdepth_rate=0.0, p_dropout_logits=0.0, use_se=False,
                 use_logit_bias=False, r_shuffle=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_2d = is_2d
        self.block = block
        self.z_to_xy_ratio = z_to_xy_ratio
        self.n_blocks_list = n_blocks_list
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.norm = norm
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.p_dropout_logits = p_dropout_logits
        self.bottleneck_ratio = bottleneck_ratio
        self.stochdepth_rate = stochdepth_rate
        self.use_se = use_se
        self.use_logit_bias = use_logit_bias
        self.r_shuffle = r_shuffle
        # we double the amount of channels every downsampling step
        # up to a max of filters_max
        self.n_stages = len(n_blocks_list)
        assert self.n_blocks_list[0] == 1, 'The stem version requires only one block on top'
        self.filters_list = [min([self.filters*2**i, self.filters_max])
                             for i in range(self.n_stages)]

        # first let's make the lists for the blocks on both pathes
        # number of input and output channels of the blocks on the contracting path
        self.in_channels_down_list = [self.in_channels, self.r_shuffle**2*self.in_channels] + self.filters_list[1:-1]
        self.in_channels_up_list = [2*f for f in self.filters_list[:-1]]
        self.out_channels_list = self.filters_list
        self.z_to_xy_ratio_list = [max([self.z_to_xy_ratio / 2**i, 1])
                                   for i in range(self.n_stages)]
        if self.is_2d:
            self.kernel_sizes_up = (self.n_stages-1) * [3]
        else:
            self.kernel_sizes_up = [(1, 3, 3) if z_to_xy >= 2 else 3 for z_to_xy in 
                                    self.z_to_xy_ratio_list]
        self.init_stride_list = [1, 1] + [get_stride(ks) for ks in self.kernel_sizes_up[1:]]


        # blocks on the contracting path
        self.first_shuffle = PixelUnshuffle3d(self.r_shuffle)
        
        self.blocks_down = []
        for n_blocks, in_ch, out_ch, init_stride, z_to_xy in zip(self.n_blocks_list[1:],
                                                                 self.in_channels_down_list[1:],
                                                                 self.out_channels_list[1:],
                                                                 self.init_stride_list[1:],
                                                                 self.z_to_xy_ratio_list[1:]):
            self.blocks_down.append(stackedResBlocks(block=self.block,
                                                     n_blocks=n_blocks,
                                                     in_channels=in_ch,
                                                     out_channels=out_ch,
                                                     init_stride=init_stride,
                                                     is_2d=self.is_2d,
                                                     z_to_xy_ratio=z_to_xy,
                                                     conv_params=self.conv_params,
                                                     norm=self.norm,
                                                     norm_params=self.norm_params,
                                                     nonlin_params=self.nonlin_params,
                                                     bottleneck_ratio=self.bottleneck_ratio,
                                                     stochdepth_rate=self.stochdepth_rate,
                                                     use_se=self.use_se))

        # blocks on the upsampling path
        self.blocks_up = []
        for in_channels, out_channels, kernel_size in zip(self.in_channels_up_list[1:],
                                                          self.out_channels_list[1:],
                                                          self.kernel_sizes_up[1:]):
            block = ConvNormNonlinBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        is_2d=self.is_2d,
                                        kernel_size=kernel_size,
                                        conv_params=self.conv_params,
                                        norm=self.norm,
                                        norm_params=self.norm_params,
                                        nonlin_params=self.nonlin_params)
            self.blocks_up.append(block)
        
        
        self.last_shuffle = PixelShuffle3d(self.r_shuffle)

        # upsaplings
        self.upsamplings = []
        for in_channels, out_channels, kernel_size in zip(self.out_channels_list[2:],
                                                          self.out_channels_list[1:-1],
                                                          self.kernel_sizes_up[1:]):
            self.upsamplings.append(UpConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           is_2d=self.is_2d,
                                           kernel_size=get_stride(kernel_size)))
        # logits
        self.all_logits = [Logits(in_channels=self.out_channels_list[1],
                                          out_channels=self.out_channels * self.r_shuffle**2,
                                          is_2d=self.is_2d,
                                          p_dropout=self.p_dropout_logits,
                                          use_bias=self.use_logit_bias)]
        for in_channels in self.out_channels_list[1:]:
            self.all_logits.append(Logits(in_channels=in_channels,
                                          out_channels=self.out_channels,
                                          is_2d=self.is_2d,
                                          p_dropout=self.p_dropout_logits,
                                          use_bias=self.use_logit_bias))

        # now important let's turn everything into a module list
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upsamplings = nn.ModuleList(self.upsamplings)
        self.all_logits = nn.ModuleList(self.all_logits)

    def forward(self, xb):
        # keep all out tensors from the contracting path
        
        xb = self.first_shuffle(xb)
        
        xb_list = []
        logs_list = []
        # contracting path
        for block in self.blocks_down[:-1]:
            xb = block(xb)
            xb_list.append(xb)

        # bottom block
        xb = self.blocks_down[-1](xb)
        logs_list.append(self.all_logits[-1](xb))

        # expanding path with logits
        for i in range(self.n_stages - 3, -1, -1):
            xb = self.upsamplings[i](xb)
            xb = torch.cat([xb, xb_list[i]], 1)
            del xb_list[i]
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i+1](xb)
            logs_list.append(logs)

        logs_list.append(self.last_shuffle(self.all_logits[0](xb)))        
        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]

    def update_prg_trn(self, param_dict, h, indx=None):
        if 'p_dropout_logits' in param_dict:
            p = (1 - h) * param_dict['p_dropout_logits'][0] + h * param_dict['p_dropout_logits'][1]
            for l in self.all_logits:
                l.dropout.p = p



# %%
class UNetResEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, z_to_xy_ratio, block='res',
                 n_blocks_list=[1, 2, 6, 3], filters=32, filters_max=384,
                 conv_params=None, norm=None, norm_params=None, nonlin_params=None,  
                 bottleneck_ratio=2, stochdepth_rate=0.0, p_dropout_logits=0.0, use_se=False,
                 use_logit_bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_2d = is_2d
        self.block = block
        self.z_to_xy_ratio = z_to_xy_ratio
        self.n_blocks_list = n_blocks_list
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.norm = norm
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.p_dropout_logits = p_dropout_logits
        self.stochdepth_rate = stochdepth_rate
        self.bottleneck_ratio = bottleneck_ratio
        self.use_se = use_se
        self.use_logit_bias = use_logit_bias
        # we double the amount of channels every downsampling step
        # up to a max of filters_max
        self.n_stages = len(n_blocks_list)
        self.filters_list = [min([self.filters*2**i, self.filters_max])
                             for i in range(self.n_stages)]

        # first let's make the lists for the blocks on both pathes
        # number of input and output channels of the blocks on the contracting path
        self.in_channels_down_list = [self.in_channels] + self.filters_list[:-1]
        self.in_channels_up_list = [2*f for f in self.filters_list[:-1]]
        self.out_channels_list = self.filters_list
        self.z_to_xy_ratio_list = [max([self.z_to_xy_ratio / 2**i, 1])
                                   for i in range(self.n_stages)]
        if self.is_2d:
            self.kernel_sizes_up = (self.n_stages-1) * [3]
        else:
            self.kernel_sizes_up = [(1, 3, 3) if z_to_xy >= 2 else 3 for z_to_xy in 
                                    self.z_to_xy_ratio_list]
        self.init_stride_list = [1] + [get_stride(ks) for ks in self.kernel_sizes_up]


        # blocks on the contracting path
        self.blocks_down = []
        if self.n_blocks_list[0] == 1:
            # we will do the first block slightly different: we're not using a residual block here
            # as the skip connection might introduce additional memory
            self.blocks_down.append(ConvNormNonlinBlock(in_channels=self.in_channels,
                                                        out_channels=self.filters,
                                                        is_2d=self.is_2d,
                                                        kernel_size=3 if self.z_to_xy_ratio < 2 else (1, 3, 3),
                                                        first_stride=1,
                                                        conv_params=self.conv_params,
                                                        norm=self.norm,
                                                        norm_params=self.norm_params,
                                                        nonlin_params=self.nonlin_params))
            i_start = 1
        else:
            i_start = 0

        for n_blocks, in_ch, out_ch, init_stride, z_to_xy in zip(self.n_blocks_list[i_start:],
                                                                 self.in_channels_down_list[i_start:],
                                                                 self.out_channels_list[i_start:],
                                                                 self.init_stride_list[i_start:],
                                                                 self.z_to_xy_ratio_list[i_start:]):
            self.blocks_down.append(stackedResBlocks(block=self.block,
                                                     n_blocks=n_blocks,
                                                     in_channels=in_ch,
                                                     out_channels=out_ch,
                                                     init_stride=init_stride,
                                                     is_2d=self.is_2d,
                                                     z_to_xy_ratio=z_to_xy,
                                                     conv_params=self.conv_params,
                                                     norm=self.norm,
                                                     norm_params=self.norm_params,
                                                     nonlin_params=self.nonlin_params,
                                                     bottleneck_ratio=self.bottleneck_ratio,
                                                     stochdepth_rate=self.stochdepth_rate,
                                                     use_se=self.use_se))

        # blocks on the upsampling path
        self.blocks_up = []
        for in_channels, out_channels, kernel_size in zip(self.in_channels_up_list,
                                                          self.out_channels_list,
                                                          self.kernel_sizes_up):
            block = ConvNormNonlinBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        is_2d=self.is_2d,
                                        kernel_size=kernel_size,
                                        conv_params=self.conv_params,
                                        norm=self.norm,
                                        norm_params=self.norm_params,
                                        nonlin_params=self.nonlin_params)
            self.blocks_up.append(block)

        # upsaplings
        self.upsamplings = []
        for in_channels, out_channels, kernel_size in zip(self.out_channels_list[1:],
                                                          self.out_channels_list[:-1],
                                                          self.kernel_sizes_up):
            self.upsamplings.append(UpConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           is_2d=self.is_2d,
                                           kernel_size=get_stride(kernel_size)))
        # logits
        self.all_logits = []
        for in_channels in self.out_channels_list:
            self.all_logits.append(Logits(in_channels=in_channels,
                                          out_channels=self.out_channels,
                                          is_2d=self.is_2d,
                                          p_dropout=self.p_dropout_logits,
                                          use_bias=self.use_logit_bias))

        # now important let's turn everything into a module list
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upsamplings = nn.ModuleList(self.upsamplings)
        self.all_logits = nn.ModuleList(self.all_logits)

    def forward(self, xb):
        # keep all out tensors from the contracting path
        xb_list = []
        logs_list = []
        # contracting path
        for block in self.blocks_down[:-1]:
            xb = block(xb)
            xb_list.append(xb)

        # bottom block
        xb = self.blocks_down[-1](xb)
        logs_list.append(self.all_logits[-1](xb))

        # expanding path with logits
        for i in range(self.n_stages - 2, -1, -1):
            xb = self.upsamplings[i](xb)
            xb = torch.cat([xb, xb_list[i]], 1)
            del xb_list[i]
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i](xb)
            logs_list.append(logs)

        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]

    def update_prg_trn(self, param_dict, h, indx=None):
        if 'p_dropout_logits' in param_dict:
            p = (1 - h) * param_dict['p_dropout_logits'][0] + h * param_dict['p_dropout_logits'][1]
            for l in self.all_logits:
                l.dropout.p = p

    def load_matching_weights_from_pretrained_model(self, data_name, preprocessed_name, model_name,
                                                    fold, model_params_name='model_parameters',
                                                    network_name='network'):
        model_CV_path = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', data_name,
                                     preprocessed_name, model_name)
        model_path = os.path.join(model_CV_path, 'fold_{}'.format(fold))
        model_params = pickle.load(open(os.path.join(model_CV_path,
                                                     model_params_name+'.pkl'),
                                        'rb'))
        net_params = model_params['network']
        # if this entry of the model paramters is wrong, the model has the wrong 
        # architecture
        assert model_params['architecture'] == 'unetresencoder'
        # create the network and load the weights
        print('Creating network to load from')
        net = UNetResEncoder(**net_params)
        path_to_weights = os.path.join(model_path, network_name+'_weights')
        print('load weights...')
        net.load_state_dict(torch.load(path_to_weights,
                                          map_location=torch.device('cpu')))

        # now we iterate over all module lists and try to transfere the weights
        print('start transfering weights')
        transfered, not_transfered = 0, 0
        for m1, m2 in zip([self.blocks_down, self.blocks_up, self.upsamplings, self.all_logits],
                          [net.blocks_down, net.blocks_up, net.upsamplings, net.all_logits]):
            for b1, b2 in zip(m1, m2):
                # iterate over all blocks
                if isinstance(b1, stackedResBlocks):
                    for rb1, rb2 in zip(b1.res_blocks, b2.res_blocks):
                        for c1, c2 in zip(b1.children(), b2.children()):
                            try:
                                c1.load_state_dict(c2.state_dict())
                                transfered += 1
                            except RuntimeError:
                                not_transfered += 1
                else:
                    for c1, c2 in zip(b1.children(), b2.children()):
                        try:
                            c1.load_state_dict(c2.state_dict())
                            transfered += 1
                        except RuntimeError:
                            not_transfered += 1

        print('Done! Loaded {} and skipped {} modules'.format(transfered, not_transfered))

# %%
class UNetResEncoderV2(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, z_to_xy_ratio,
                 n_blocks_list=[1, 2, 6, 3], filters=32, filters_max=384,
                 conv_params=None, norm=None, norm_params=None, nonlin_params=None,
                 use_5x5_on_full_res=True, use_logit_bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_2d = is_2d
        self.z_to_xy_ratio = z_to_xy_ratio
        self.n_blocks_list = n_blocks_list
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.norm = norm
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.use_5x5_on_full_res = use_5x5_on_full_res
        self.use_logit_bias = use_logit_bias
        
        
        self.block = 'res'
        # we double the amount of channels every downsampling step
        # up to a max of filters_max
        self.n_stages = len(n_blocks_list)
        
        if np.isscalar(self.filters):
            self.filters_list = [min([self.filters*2**i, self.filters_max])
                                 for i in range(self.n_stages)]
        else:
            self.filters_list = self.filters

        # first let's make the lists for the blocks on both pathes
        # number of input and output channels of the blocks on the contracting path
        self.in_channels_down_list = [self.in_channels] + self.filters_list[:-1]
        self.in_channels_up_list = [2*f for f in self.filters_list[:-1]]
        self.out_channels_list = self.filters_list
        self.z_to_xy_ratio_list = [max([self.z_to_xy_ratio / 2**i, 1])
                                   for i in range(self.n_stages)]
        if self.is_2d:
            self.kernel_sizes_up = (self.n_stages-1) * [3]
        else:
            self.kernel_sizes_up = [(1, 3, 3) if z_to_xy >= 2 else 3 for z_to_xy in 
                                    self.z_to_xy_ratio_list]
        self.init_stride_list = [1] + [get_stride(ks) for ks in self.kernel_sizes_up]


        # blocks on the contracting path
        self.blocks_down = []
        if self.n_blocks_list[0] == 1:
            # we will do the first block slightly different: we're not using a residual block here
            # as the skip connection might introduce additional memory during inference
            
            if self.use_5x5_on_full_res:
            
                if self.z_to_xy_ratio < 2:
                    kernel_size = 5
                elif self.z_to_xy_ratio < 4:
                    kernel_size = (3, 5, 5)
                else:
                    kernel_size = (1, 5, 5)
                    
            
                self.blocks_down.append(ConvNormNonlinBlockV2(in_channels=self.in_channels,
                                                            out_channels=self.filters_list[0],
                                                            is_2d=self.is_2d,
                                                            kernel_size=kernel_size,
                                                            first_stride=1,
                                                            conv_params=self.conv_params,
                                                            norm=self.norm,
                                                            norm_params=self.norm_params,
                                                            nonlin_params=self.nonlin_params))
            else:
                self.blocks_down.append(ConvNormNonlinBlock(in_channels=self.in_channels,
                                                            out_channels=self.filters_list[0],
                                                            is_2d=self.is_2d,
                                                            kernel_size=3 if self.z_to_xy_ratio < 2 else (1, 3, 3),
                                                            first_stride=1,
                                                            conv_params=self.conv_params,
                                                            norm=self.norm,
                                                            norm_params=self.norm_params,
                                                            nonlin_params=self.nonlin_params))
            i_start = 1
        else:
            i_start = 0

        for n_blocks, in_ch, out_ch, init_stride, z_to_xy in zip(self.n_blocks_list[i_start:],
                                                                 self.in_channels_down_list[i_start:],
                                                                 self.out_channels_list[i_start:],
                                                                 self.init_stride_list[i_start:],
                                                                 self.z_to_xy_ratio_list[i_start:]):
            self.blocks_down.append(stackedResBlocks(block=self.block,
                                                     n_blocks=n_blocks,
                                                     in_channels=in_ch,
                                                     out_channels=out_ch,
                                                     init_stride=init_stride,
                                                     is_2d=self.is_2d,
                                                     z_to_xy_ratio=z_to_xy,
                                                     conv_params=self.conv_params,
                                                     norm=self.norm,
                                                     norm_params=self.norm_params,
                                                     nonlin_params=self.nonlin_params))

        # blocks on the upsampling path
        self.blocks_up = []
        
        if self.use_5x5_on_full_res:
            
            block = ConvNormNonlinBlockV2(in_channels=self.in_channels_up_list[0],
                                          out_channels=self.out_channels_list[0],
                                          is_2d=self.is_2d,
                                          kernel_size=kernel_size,
                                          conv_params=self.conv_params,
                                          norm=self.norm,
                                          norm_params=self.norm_params,
                                          nonlin_params=self.nonlin_params)
            self.blocks_up.append(block)
            j_start = 1
        else:
            j_start = 0
        
        for in_channels, out_channels, kernel_size in zip(self.in_channels_up_list[j_start:],
                                                          self.out_channels_list[j_start:],
                                                          self.kernel_sizes_up[j_start:]):
            block = ConvNormNonlinBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        is_2d=self.is_2d,
                                        kernel_size=kernel_size,
                                        conv_params=self.conv_params,
                                        norm=self.norm,
                                        norm_params=self.norm_params,
                                        nonlin_params=self.nonlin_params)
            self.blocks_up.append(block)

        # upsaplings
        self.upsamplings = []
        for in_channels, out_channels, kernel_size in zip(self.out_channels_list[1:],
                                                          self.out_channels_list[:-1],
                                                          self.kernel_sizes_up):
            self.upsamplings.append(UpConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           is_2d=self.is_2d,
                                           kernel_size=get_stride(kernel_size)))
        # logits
        self.all_logits = []
        for in_channels in self.out_channels_list:
            self.all_logits.append(Logits(in_channels=in_channels,
                                          out_channels=self.out_channels,
                                          is_2d=self.is_2d,
                                          use_bias=self.use_logit_bias))

        # now important let's turn everything into a module list
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upsamplings = nn.ModuleList(self.upsamplings)
        self.all_logits = nn.ModuleList(self.all_logits)

    def forward(self, xb):
        # keep all out tensors from the contracting path
        xb_list = []
        logs_list = []
        # contracting path
        for block in self.blocks_down[:-1]:
            xb = block(xb)
            xb_list.append(xb)

        # bottom block
        xb = self.blocks_down[-1](xb)
        logs_list.append(self.all_logits[-1](xb))

        # expanding path with logits
        for i in range(self.n_stages - 2, -1, -1):
            xb = self.upsamplings[i](xb)
            xb = torch.cat([xb, xb_list[i]], 1)
            del xb_list[i]
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i](xb)
            logs_list.append(logs)

        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]


# %%
class UResNet(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, z_to_xy_ratio, block='res',
                 n_blocks_list=[4, 2, 1], filters=32, filters_max=384,
                 conv_params=None, norm=None, norm_params=None, nonlin_params=None,  
                 bottleneck_ratio=2, stochdepth_rate=0.0, p_dropout_logits=0.0, use_se=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_2d = is_2d
        self.block = block
        self.z_to_xy_ratio = z_to_xy_ratio
        self.n_blocks_list = n_blocks_list
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.norm = norm
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.p_dropout_logits = p_dropout_logits
        self.stochdepth_rate = stochdepth_rate
        self.bottleneck_ratio = bottleneck_ratio
        self.use_se = use_se
        # we double the amount of channels every downsampling step
        # up to a max of filters_max
        self.n_stages = len(n_blocks_list)
        self.filters_list = [min([self.filters*2**i, self.filters_max])
                             for i in range(self.n_stages)]

        # first let's make the lists for the blocks on both pathes
        # number of input and output channels of the blocks on the contracting path
        self.in_channels_down_list = [self.in_channels] + self.filters_list[:-1]
        self.in_channels_up_list = [2*f for f in self.filters_list[:-1]]
        self.out_channels_list = self.filters_list
        self.z_to_xy_ratio_list = [max([self.z_to_xy_ratio / 2**i, 1])
                                   for i in range(self.n_stages)]
        if self.is_2d:
            self.kernel_sizes_up = (self.n_stages-1) * [3]
        else:
            self.kernel_sizes_up = [(1, 3, 3) if z_to_xy >= 2 else 3 for z_to_xy in 
                                    self.z_to_xy_ratio_list]
        self.init_stride_list = [1] + [get_stride(ks) for ks in self.kernel_sizes_up]


        # blocks on the contracting path
        self.blocks_down = []
        for n_blocks, in_ch, out_ch, init_stride, z_to_xy in zip(self.n_blocks_list,
                                                                 self.in_channels_down_list,
                                                                 self.out_channels_list,
                                                                 self.init_stride_list,
                                                                 self.z_to_xy_ratio_list):
            self.blocks_down.append(stackedResBlocks(block=self.block,
                                                     n_blocks=n_blocks,
                                                     in_channels=in_ch,
                                                     out_channels=out_ch,
                                                     init_stride=init_stride,
                                                     is_2d=self.is_2d,
                                                     z_to_xy_ratio=z_to_xy,
                                                     conv_params=self.conv_params,
                                                     norm=self.norm,
                                                     norm_params=self.norm_params,
                                                     nonlin_params=self.nonlin_params,
                                                     bottleneck_ratio=self.bottleneck_ratio,
                                                     stochdepth_rate=self.stochdepth_rate,
                                                     use_se=self.use_se))

        # blocks on the upsampling path
        self.blocks_up = []
        for n_blocks, in_ch, out_ch, z_to_xy in zip(self.n_blocks_list[:-1],
                                                    self.in_channels_up_list,
                                                    self.out_channels_list[:-1],
                                                    self.z_to_xy_ratio_list[:-1]):
            self.blocks_up.append(stackedResBlocks(block=self.block,
                                                   n_blocks=n_blocks,
                                                   in_channels=in_ch,
                                                   out_channels=out_ch,
                                                   init_stride=1,
                                                   is_2d=self.is_2d,
                                                   z_to_xy_ratio=z_to_xy,
                                                   conv_params=self.conv_params,
                                                   norm=self.norm,
                                                   norm_params=self.norm_params,
                                                   nonlin_params=self.nonlin_params,
                                                   bottleneck_ratio=self.bottleneck_ratio,
                                                   stochdepth_rate=self.stochdepth_rate,
                                                   use_se=self.use_se))

        # upsaplings
        self.upsamplings = []
        for in_channels, out_channels, kernel_size in zip(self.out_channels_list[1:],
                                                          self.out_channels_list[:-1],
                                                          self.kernel_sizes_up):
            self.upsamplings.append(UpConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           is_2d=self.is_2d,
                                           kernel_size=get_stride(kernel_size)))
        # logits
        self.all_logits = []
        for in_channels in self.out_channels_list:
            self.all_logits.append(Logits(in_channels=in_channels,
                                          out_channels=self.out_channels,
                                          is_2d=self.is_2d,
                                          p_dropout=self.p_dropout_logits))

        # now important let's turn everything into a module list
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upsamplings = nn.ModuleList(self.upsamplings)
        self.all_logits = nn.ModuleList(self.all_logits)

    def forward(self, xb):
        # keep all out tensors from the contracting path
        xb_list = []
        logs_list = []
        # contracting path
        for block in self.blocks_down[:-1]:
            xb = block(xb)
            xb_list.append(xb)

        # bottom block
        xb = self.blocks_down[-1](xb)
        logs_list.append(self.all_logits[-1](xb))

        # expanding path with logits
        for i in range(self.n_stages - 2, -1, -1):
            xb = self.upsamplings[i](xb)
            xb = torch.cat([xb, xb_list[i]], 1)
            del xb_list[i]
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i](xb)
            logs_list.append(logs)

        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]

    def update_prg_trn(self, param_dict, h, indx=None):
        if 'p_dropout_logits' in param_dict:
            p = (1 - h) * param_dict['p_dropout_logits'][0] + h * param_dict['p_dropout_logits'][1]
            for l in self.all_logits:
                l.dropout.p = p

    def load_matching_weights_from_pretrained_model(self, data_name, preprocessed_name, model_name,
                                                    fold, model_params_name='model_parameters',
                                                    network_name='network'):
        model_CV_path = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', data_name,
                                     preprocessed_name, model_name)
        model_path = os.path.join(model_CV_path, 'fold_{}'.format(fold))
        model_params = pickle.load(open(os.path.join(model_CV_path,
                                                     model_params_name+'.pkl'),
                                        'rb'))
        net_params = model_params['network']
        # if this entry of the model paramters is wrong, the model has the wrong 
        # architecture
        assert model_params['architecture'] == 'unetresencoder'
        # create the network and load the weights
        print('Creating network to load from')
        net = UNetResEncoder(**net_params)
        path_to_weights = os.path.join(model_path, network_name+'_weights')
        print('load weights...')
        net.load_state_dict(torch.load(path_to_weights,
                                          map_location=torch.device('cpu')))

        # now we iterate over all module lists and try to transfere the weights
        print('start transfering weights')
        transfered, not_transfered = 0, 0
        for m1, m2 in zip([self.blocks_down, self.blocks_up, self.upsamplings, self.all_logits],
                          [net.blocks_down, net.blocks_up, net.upsamplings, net.all_logits]):
            for b1, b2 in zip(m1, m2):
                # iterate over all blocks
                if isinstance(b1, stackedResBlocks):
                    for rb1, rb2 in zip(b1.res_blocks, b2.res_blocks):
                        for c1, c2 in zip(b1.children(), b2.children()):
                            try:
                                c1.load_state_dict(c2.state_dict())
                                transfered += 1
                            except RuntimeError:
                                not_transfered += 1
                else:
                    for c1, c2 in zip(b1.children(), b2.children()):
                        try:
                            c1.load_state_dict(c2.state_dict())
                            transfered += 1
                        except RuntimeError:
                            not_transfered += 1

        print('Done! Loaded {} and skipped {} modules'.format(transfered, not_transfered))


# %%
if __name__ == '__main__':
    net = UNetResShuffleEncoder(in_channels=1,
                                out_channels=2,
                                is_2d=False,
                                z_to_xy_ratio=8,
                                filters=4).cuda()
    
    xb = torch.zeros((1, 1, 32, 256, 256)).cuda()
    logs_list = net(xb)
    