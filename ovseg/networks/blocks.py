import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

GAMMA_RELU = 1.7128585504496627  # =np.sqrt(2 / (1 - 1/np.pi))


# %% nonlinearities

class scaledReLU(nn.ReLU):
    # TODO integrate the scale here so that we can put the alpha and beta in here
    def forward(self, input):
        return GAMMA_RELU * super().forward(input)


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


# %% weight standardized convolutions
class WSConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0,
                 dilation=1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)
        self.unbiased = kernel_size != 1
        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer('fan_in', torch.tensor(self.weight.shape[1:].numel(),
                                                    requires_grad=False).type_as(self.weight),
                             persistent=False)

    def standardized_weights(self):
        # Original code: HWCN
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True, unbiased=self.unbiased)
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale

    def forward(self, xb):
        return F.conv2d(
            input=xb,
            weight=self.standardized_weights(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


class WSConv3d(nn.Conv3d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0,
                 dilation=1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)
        self.unbiased = kernel_size != 1
        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer('fan_in', torch.tensor(self.weight.shape[1:].numel(),
                                                    requires_grad=False).type_as(self.weight),
                             persistent=False)

    def standardized_weights(self):
        # Original code: HWCN
        mean = torch.mean(self.weight, axis=[1, 2, 3, 4], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3, 4], keepdims=True, unbiased=self.unbiased)
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale

    def forward(self, xb):
        return F.conv3d(
            input=xb,
            weight=self.standardized_weights(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


# %%
class StochDepth(nn.Module):
    def __init__(self, stochdepth_rate: float):
        super().__init__()

        self.stochdepth_rate = stochdepth_rate

    def forward(self, xb):
        if not self.training:
            return xb

        batch_size = xb.shape[0]
        ones = [1 for _ in range(len(xb.shape) - 1)]
        rand_tensor = torch.rand(batch_size, *ones).type_as(xb).to(xb.device)
        keep_prob = 1 - self.stochdepth_rate
        binary_tensor = torch.floor(rand_tensor + keep_prob)

        return xb * binary_tensor


# %%
class SE_unit(nn.Module):

    def __init__(self, num_channels, reduction=4, is_2d=False):
        super().__init__()
        if is_2d:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(num_channels, num_channels // reduction)
        self.fc2 = nn.Linear(num_channels // reduction, num_channels)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        self.nonlin1 = scaledReLU()
        self.nonlin2 = nn.Sigmoid()

    def forward(self, xb):

        bs, n_ch = xb.shape[0], xb.shape[1]
        ones = [1 for _ in range(len(xb.shape) - 2)]
        xb_se = self.avg_pool(xb).view(bs, n_ch)
        xb_se = self.nonlin1(self.fc1(xb_se))
        xb_se = 2.0 * self.nonlin2(self.fc2(xb_se)).view((bs, n_ch, *ones))

        return xb * xb_se


# %% normalization free Blocks
class nfConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hid_channels=None, is_2d=False,
                 kernel_size=3, first_stride=1, conv_params=None, nonlin_params=None,
                 is_inference_block=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = get_padding(self.kernel_size)
        self.first_stride = first_stride
        self.hid_channels = out_channels if hid_channels is None else hid_channels
        self.is_2d = is_2d
        self.conv_params = conv_params
        self.nonlin_params = nonlin_params
        self.is_inference_block = is_inference_block

        if self.conv_params is None:
            self.conv_params = {'bias': True}
        if self.nonlin_params is None:
            self.nonlin_params = {'inplace': True}
        # init convolutions, normalisation and nonlinearities
        if self.is_2d:
            if self.is_inference_block:
                conv_fctn = nn.Conv2d
            else:
                conv_fctn = WSConv2d
        else:
            if self.is_inference_block:
                conv_fctn = nn.Conv3d
            else:
                conv_fctn = WSConv3d

        self.conv1 = conv_fctn(self.in_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               stride=self.first_stride, **self.conv_params)
        self.conv2 = conv_fctn(self.out_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               **self.conv_params)
        if self.is_inference_block:
            self.nonlin1 = nn.ReLU(**self.nonlin_params)
            self.nonlin2 = nn.ReLU(**self.nonlin_params)
        else:
            self.nonlin1 = scaledReLU(**self.nonlin_params)
            self.nonlin2 = scaledReLU(**self.nonlin_params)

    def forward(self, xb):
        xb = self.conv1(xb)
        xb = self.nonlin1(xb)
        xb = self.conv2(xb)
        xb = self.nonlin2(xb)
        return xb


class nfConvResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d=False, alpha=0.2, beta=1,
                 kernel_size=3, first_stride=1, conv_params=None, nonlin_params=None,
                 se_reduction=4, stochdepth_rate=0, is_inference_block=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.kernel_size = kernel_size
        self.padding = get_padding(self.kernel_size)
        self.first_stride = first_stride
        self.is_2d = is_2d
        self.conv_params = conv_params
        self.nonlin_params = nonlin_params
        self.se_reduction = se_reduction
        self.stochdepth_rate = stochdepth_rate
        self.use_stochdepth = stochdepth_rate > 0
        self.is_inference_block = is_inference_block
        if self.is_inference_block:
            raise NotImplementedError('Inference blocks have not been implemented for residual '
                                      'blocks')

        if self.conv_params is None:
            self.conv_params = {'bias': True}
        if self.nonlin_params is None:
            self.nonlin_params = {'inplace': True}
        # init convolutions, normalisation and nonlinearities
        if self.is_2d:
            conv_fctn = WSConv2d
            pool_fctn = nn.AvgPool2d
        else:
            conv_fctn = WSConv3d
            pool_fctn = nn.AvgPool3d

        self.tau = nn.Parameter(torch.zeros(()))
        self.conv1 = conv_fctn(self.in_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               stride=self.first_stride, **self.conv_params)
        self.conv2 = conv_fctn(self.out_channels, self.out_channels,
                               self.kernel_size, padding=self.padding,
                               **self.conv_params)
        self.se = SE_unit(self.out_channels, self.se_reduction, is_2d=is_2d)
        self.stochdepth = StochDepth(self.stochdepth_rate)
        self.nonlin1 = scaledReLU(**self.nonlin_params)
        self.nonlin2 = scaledReLU(**self.nonlin_params)

        self.downsample = np.prod(self.first_stride) > 1
        if self.downsample and self.in_channels == self.out_channels:
            self.skip = pool_fctn(self.first_stride, self.first_stride)
        elif self.downsample and self.in_channels != self.out_channels:
            self.skip = nn.Sequential(pool_fctn(self.first_stride, self.first_stride),
                                      conv_fctn(self.in_channels, self.out_channels, 1))
        elif not self.downsample and self.in_channels == self.out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = conv_fctn(self.in_channels, self.out_channels, 1)

    def forward(self, xb):
        skip = self.skip(xb)
        xb = self.nonlin1(xb * self.beta)
        xb = self.conv1(xb)
        xb = self.nonlin2(xb)
        xb = self.conv2(xb)
        xb = self.se(xb)
        if self.use_stochdepth:
            xb = self.stochdepth(xb)
        xb = xb * self.alpha * self.tau
        return xb + skip


class nfBottleneckConvResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d=False, alpha=0.2, beta=1, kernel_size=3,
                 first_stride=1, conv_params=None, nonlin_params=None, bottleneck_ratio=2,
                 se_reduction=4, stochdepth_rate=0, is_inference_block=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.kernel_size = kernel_size
        self.padding = get_padding(self.kernel_size)
        self.first_stride = first_stride
        self.is_2d = is_2d
        self.conv_params = conv_params
        self.nonlin_params = nonlin_params
        self.bottleneck_ratio = bottleneck_ratio
        self.se_reduction = se_reduction
        self.stochdepth_rate = stochdepth_rate
        self.use_stochdepth = stochdepth_rate > 0
        self.is_inference_block = is_inference_block
        if self.is_inference_block:
            raise NotImplementedError('Inference blocks have not been implemented for residual '
                                      'blocks')
        if self.conv_params is None:
            self.conv_params = {'bias': True}
        if self.nonlin_params is None:
            self.nonlin_params = {'inplace': True}
        # init convolutions, normalisation and nonlinearities
        if self.is_2d:
            conv_fctn = WSConv2d
            pool_fctn = nn.AvgPool2d
        else:
            conv_fctn = WSConv3d
            pool_fctn = nn.AvgPool3d

        self.tau = nn.Parameter(torch.zeros(()))
        self.conv1 = conv_fctn(self.in_channels,
                               self.out_channels // self.bottleneck_ratio,
                               kernel_size=1, **self.conv_params)
        self.conv2 = conv_fctn(self.out_channels // self.bottleneck_ratio,
                               self.out_channels // self.bottleneck_ratio,
                               self.kernel_size, padding=self.padding,
                               stride=self.first_stride, **self.conv_params)
        self.conv3 = conv_fctn(self.out_channels // self.bottleneck_ratio,
                               self.out_channels // self.bottleneck_ratio,
                               self.kernel_size, padding=self.padding,
                               **self.conv_params)
        self.conv4 = conv_fctn(self.out_channels // self.bottleneck_ratio,
                               self.out_channels,
                               kernel_size=1, **self.conv_params)
        self.se = SE_unit(self.out_channels, self.se_reduction, is_2d=is_2d)
        self.stochdepth = StochDepth(self.stochdepth_rate)

        self.nonlin1 = scaledReLU(**self.nonlin_params)
        self.nonlin2 = scaledReLU(**self.nonlin_params)
        self.nonlin3 = scaledReLU(**self.nonlin_params)
        self.nonlin4 = scaledReLU(**self.nonlin_params)

        self.downsample = np.prod(self.first_stride) > 1
        if self.downsample and self.in_channels == self.out_channels:
            self.skip = pool_fctn(self.first_stride, self.first_stride)
        elif self.downsample and self.in_channels != self.out_channels:
            self.skip = nn.Sequential(pool_fctn(self.first_stride, self.first_stride),
                                      conv_fctn(self.in_channels, self.out_channels, 1))
        elif not self.downsample and self.in_channels == self.out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = conv_fctn(self.in_channels, self.out_channels, 1)

    def forward(self, xb):
        skip = self.skip(xb)
        xb = xb * self.beta
        xb = self.nonlin1(xb)
        xb = self.conv1(xb)
        xb = self.nonlin2(xb)
        xb = self.conv2(xb)
        xb = self.nonlin3(xb)
        xb = self.conv3(xb)
        xb = self.nonlin4(xb)
        xb = self.conv4(xb)
        xb = self.se(xb)
        if self.use_stochdepth:
            xb = self.stochdepth(xb)
        xb = xb * self.alpha * self.tau
        return xb + skip


# %% concatination of residual blocks
class nfConvResStage(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d=False, n_blocks=1,
                 alpha=0.2, kernel_size=3,
                 first_stride=1, conv_params=None, nonlin_params=None, use_bottleneck=False,
                 bottleneck_ratio=0.5, se_reduction=4, stochdepth_rate=0):
        super().__init__()
        first_stride_list = [first_stride] + [1 for _ in range(n_blocks-1)]
        in_channels_list = [in_channels] + [out_channels for _ in range(n_blocks-1)]
        blocks_list = []
        expected_std = 1.0
        if use_bottleneck:
            for first_stride, in_channels in zip(first_stride_list, in_channels_list):
                # we use a slightly different definition from the paper to execute the
                # multiplication instead of a division
                beta = 1.0/expected_std
                blocks_list.append(nfBottleneckConvResBlock(in_channels, out_channels, is_2d,
                                                            alpha=alpha, beta=beta,
                                                            kernel_size=kernel_size,
                                                            first_stride=first_stride,
                                                            conv_params=conv_params,
                                                            nonlin_params=nonlin_params,
                                                            bottleneck_ratio=bottleneck_ratio,
                                                            se_reduction=se_reduction,
                                                            stochdepth_rate=stochdepth_rate))
                # now update
                expected_std = (expected_std ** 2 + alpha ** 2)**0.5
        else:
            for first_stride, in_channels in zip(first_stride_list, in_channels_list):
                beta = 1.0/expected_std
                blocks_list.append(nfConvResBlock(in_channels, out_channels, is_2d,
                                                  alpha=alpha, beta=beta,
                                                  kernel_size=kernel_size,
                                                  first_stride=first_stride,
                                                  conv_params=conv_params,
                                                  nonlin_params=nonlin_params,
                                                  se_reduction=se_reduction,
                                                  stochdepth_rate=stochdepth_rate))
                expected_std = (expected_std ** 2 + alpha ** 2)**0.5
        self.blocks = nn.ModuleList(blocks_list)

    def forward(self, xb):
        for block in self.blocks:
            xb = block(xb)
        return xb
