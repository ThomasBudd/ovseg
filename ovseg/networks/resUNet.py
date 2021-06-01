import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ovseg.networks.blocks import StochDepth

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

# %%
class ResBlock(nn.Module):
    # keeping this class general for the 3d case

    def __init__(self, in_channels, out_channels, is_2d=False, kernel_size=3, kernel_size2=None,
                 first_stride=1, conv_params=None, norm='inst', norm_params=None,
                 nonlin_params=None, stochdepth_rate=0.2):
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

    def forward(self, xb):
        xb_res = self.nonlin2(self.norm2(self.conv2(self.nonlin1(self.norm1(self.conv1(xb))))))
        return self.skip(xb) + self.a * self.stochdepth(xb_res)


class ResBottleneckBlock(nn.Module):
    # keeping this class general for the 3d case

    def __init__(self, in_channels, out_channels, is_2d=False, kernel_size=3, kernel_size2=None,
                 first_stride=1, conv_params=None, norm='inst', norm_params=None,
                 nonlin_params=None, bottleneck_ratio=2, stochdepth_rate=0.2):
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

    def forward(self, xb):

        skip = self.skip(xb)
        xb = self.nonlin1(self.norm1(self.conv1(xb)))
        xb = self.nonlin2(self.norm2(self.conv2(xb)))
        xb = self.nonlin3(self.norm3(self.conv3(xb)))
        xb = self.nonlin4(self.norm4(self.conv4(xb)))

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
                 nonlin_params=None, bottleneck_ratio=2, stochdepth_rate=0.2):
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
        if self.block == 'bottleneck':
            self.bottleneck_ratio = bottleneck_ratio
        if not self.block == 'mergeandrun':
            self.stochdepth_rate = stochdepth_rate

        # now create the lists we need for creating the blocks
        in_channels_list = [self.in_channels] + (self.n_blocks - 1) * [self.out_channels]
        init_stride_list = [self.init_stride] + (self.n_blocks -1) * [1]

        kernel_sizes_list = 2 * self.n_blocks * [3]
        if not self.is_2d and self.z_to_xy_ratio > 1:
            # in this case we want to apply some inplane convolutions
            for i in range(1, 2*self.n_blocks + 1):
                if i % self.z_to_xy_ratio != 0:
                    kernel_sizes_list[i] = (1, 3, 3)

        res_blocks = []
        if self.block == 'res':
            res_block = ResBlock
            kwargs = {'stochdepth_rate': self.stochdepth_rate}
        elif self.block == 'bottleneck':
            res_block = ResBottleneckBlock
            kwargs = {'stochdepth_rate': self.stochdepth_rate,
                      'bottleneck_ratio': self.bottleneck_ratio}
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

# %% now simply the logits
class Logits(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, p_dropout=0):
        super().__init__()
        if is_2d:
            self.logits = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.dropout = nn.Dropout2d(p_dropout, inplace=True)
        else:
            self.logits = nn.Conv3d(in_channels, out_channels, 1, bias=False)
            self.dropout = nn.Dropout3d(p_dropout, inplace=True)
        nn.init.kaiming_normal_(self.logits.weight)

    def forward(self, xb):
        return self.dropout(self.logits(xb))


# %%
class UNetResEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, block, z_to_xy_ratio=1,
                 n_blocks_list=[1, 2, 6, 3], filters=32, filters_max=384,
                 conv_params=None, norm=None, norm_params=None, nonlin_params=None, 
                 bottleneck_ratio=2, stochdepth_rate=0.2, p_dropout_logits=0.0):
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
        self.init_stride_list = [get_stride(ks) for ks in self.kernel_sizes_up]


        # blocks on the contracting path
        self.blocks_down = []
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
        for n_blocks, in_ch, out_ch, init_stride, z_to_xy in zip(self.n_blocks_list[1:],
                                                                 self.in_channels_down_list[1:],
                                                                 self.out_channels_list[1:],
                                                                 self.init_stride_list,
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
                                                     stochdepth_rate=self.stochdepth_rate))

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


# %%
if __name__ == '__main__':
    net = UNetResEncoder(in_channels=1, out_channels=2, is_2d=False, block='bottleneck', filters=24,
                         z_to_xy_ratio=4)
    xb = torch.randn((2, 1, 32, 128, 128))
    print(net)
    yb = net(xb)