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
        return tuple([(k + 1)//2 for k in kernel_size])
    else:
        return (kernel_size + 1) // 2


# %%
class WSConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0,
                 dilation=1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)
        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer('fan_in', torch.tensor(self.weight.shape[1:].numel(),
                                                    requires_grad=False).type_as(self.weight),
                             persistent=False)

    def standardized_weights(self):
        # Original code: HWCN
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
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
        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer('fan_in', torch.tensor(self.weight.shape[1:].numel(),
                                                    requires_grad=False).type_as(self.weight),
                             persistent=False)

    def standardized_weights(self):
        # Original code: HWCN
        mean = torch.mean(self.weight, axis=[1, 2, 3, 4], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3, 4], keepdims=True)
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
        self.nonlin1 = nn.ReLU()
        self.nonlin2 = nn.Sigmoid()

    def forward(self, xb):

        bs, n_ch = xb.shape[0], xb.shape[1]
        ones = [1 for _ in range(len(xb.shape) - 2)]
        xb_se = self.avg_pool(xb).view(bs, n_ch)
        xb_se = self.nonlin1(self.fc1(xb_se))
        xb_se = 2.0 * self.nonlin2(self.fc2(xb_se)).view((bs, n_ch, *ones))

        return xb * xb_se


# %%
class nfConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d=False, alpha=0.2, beta=1,
                 kernel_size=3, first_stride=1, conv_params=None, nonlin_params=None,
                 se_reduction=4, stochdepth_rate=0):
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
        self.nonlin1 = nn.ReLU(**self.nonlin_params)
        self.nonlin2 = nn.ReLU(**self.nonlin_params)

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
        xb = self.nonlin1(xb * self.beta) * GAMMA
        xb = self.conv1(xb)
        xb = self.nonlin2(xb) * GAMMA
        xb = self.conv2(xb)
        xb = self.se(xb)
        if self.use_stochdepth:
            xb = self.stochdepth(xb)
        xb = xb * self.alpha * self.tau
        return xb + skip


class nfConvBottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d=False, alpha=0.2, beta=1, kernel_size=3,
                 first_stride=1, conv_params=None, nonlin_params=None, bottleneck_ratio=2,
                 se_reduction=4, stochdepth_rate=0):
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

        self.nonlin1 = nn.ReLU(**self.nonlin_params)
        self.nonlin2 = nn.ReLU(**self.nonlin_params)
        self.nonlin3 = nn.ReLU(**self.nonlin_params)
        self.nonlin4 = nn.ReLU(**self.nonlin_params)

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
        xb = self.nonlin1(xb) * GAMMA
        xb = self.conv1(xb)
        xb = self.nonlin2(xb) * GAMMA
        xb = self.conv2(xb)
        xb = self.nonlin3(xb) * GAMMA
        xb = self.conv3(xb)
        xb = self.nonlin4(xb) * GAMMA
        xb = self.conv4(xb)
        xb = self.se(xb)
        if self.use_stochdepth:
            xb = self.stochdepth(xb)
        xb = xb * self.alpha * self.tau
        return xb + skip


class nfConvStage(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d=False, n_blocks=1,
                 alpha=0.2, kernel_size=3,
                 first_stride=1, conv_params=None, nonlin_params=None, use_bottleneck=False,
                 se_reduction=4, stochdepth_rate=0):
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
                blocks_list.append(nfConvBottleneckBlock(in_channels, out_channels, is_2d,
                                                         alpha=alpha, beta=beta,
                                                         kernel_size=kernel_size,
                                                         first_stride=first_stride,
                                                         conv_params=conv_params,
                                                         nonlin_params=nonlin_params,
                                                         se_reduction=se_reduction,
                                                         stochdepth_rate=stochdepth_rate))
                # now update
                expected_std = (expected_std ** 2 + alpha ** 2)**0.5
        else:
            for first_stride, in_channels in zip(first_stride_list, in_channels_list):
                beta = 1.0/expected_std
                blocks_list.append(nfConvBlock(in_channels, out_channels, is_2d,
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


# %% transposed convolutions
class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d=False, kernel_size=2,
                 upsampling='conv', align_corners=True):
        super().__init__()
        assert upsampling in ['conv', 'linear']
        if is_2d:
            if upsampling == 'conv':
                self.up = nn.ConvTranspose2d(in_channels,
                                             out_channels,
                                             kernel_size,
                                             stride=kernel_size,
                                             bias=False)
                nn.init.kaiming_normal_(self.up.weight)
            elif upsampling == 'linear':
                self.conv = WSConv2d(in_channels,
                                     out_channels,
                                     kernel_size=1)
                self.interp = nn.Upsample(scale_factor=kernel_size, mode='bilinear',
                                          align_corners=align_corners)
                self.up = nn.Sequential(self.conv, self.interp)
        else:
            if upsampling == 'conv':
                self.up = nn.ConvTranspose3d(in_channels, out_channels,
                                             kernel_size, stride=kernel_size,
                                             bias=False)
                nn.init.kaiming_normal_(self.up.weight)
            elif upsampling == 'linear':
                self.conv = WSConv3d(in_channels,
                                     out_channels,
                                     kernel_size=1)
                self.interp = nn.Upsample(scale_factor=kernel_size, mode='trilinear',
                                          align_corners=align_corners)
                self.up = nn.Sequential(self.conv, self.interp)

    def forward(self, xb):
        return self.up(xb)


# %%
class concat_attention(nn.Module):

    def __init__(self, in_channels, is_2d=False):
        super().__init__()
        if is_2d:
            self.logits = nn.Conv2d(in_channels, 1, 1)
        else:
            self.logits = nn.Conv3d(in_channels, 1, 1)

        nn.init.zeros_(self.logits.bias)
        nn.init.zeros_(self.logits.weight)
        self.nonlin = torch.sigmoid

    def forward(self, xb_up, xb_skip):

        # we multiply by 2 for the same reason we do it in S&E Units
        # with zero init of the logits weights this attention gate is doing nothing
        attention = 2.0 * self.nonlin(self.logits(xb_up))
        return torch.cat([xb_up, attention * xb_skip], dim=1)


class concat(nn.Module):

    def forward(self, xb_up, xb_skip):
        return torch.cat([xb_up, xb_skip], dim=1)


# %% now simply the logits
class Logits(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d=False, dropout_rate=0):
        super().__init__()
        if is_2d:
            self.logits = nn.Conv2d(in_channels, out_channels, 1)
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.logits = nn.Conv3d(in_channels, out_channels, 1)
            self.dropout = nn.Dropout3d(dropout_rate)
        nn.init.kaiming_normal_(self.logits.weight, nonlinearity='relu')
        nn.init.zeros_(self.logits.bias)

    def forward(self, xb):
        return self.dropout(self.logits(xb))


# %%
class nfUNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes,
                 is_2d=False, n_blocks=None, filters=32, filters_max=512, n_pyramid_scales=None,
                 conv_params=None, nonlin_params=None, use_bottleneck=False, se_reduction=4,
                 use_attention_gates=False, alpha=0.2, stochdepth_rate=0, dropout_rate=0,
                 upsampling='conv', align_corners=True, factor_skip_conn=0.5):
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
            self.n_blocks = n_blocks
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.nonlin_params = nonlin_params
        self.use_bottleneck = use_bottleneck
        self.se_reduction = se_reduction
        self.use_attention_gates = use_attention_gates
        self.alpha = alpha
        self.stochdepth_rate = stochdepth_rate
        self.dropout_rate = dropout_rate
        self.upsampling = upsampling
        self.align_corners = align_corners
        self.factor_skip_conn = factor_skip_conn
        if self.factor_skip_conn >= 1 or self.factor_skip_conn < 0:
            raise ValueError('ERROR: factor_skip_conn, the factor at which the channels at the'
                             ' skip connections are reduced, must be between 0 and 1 exclusively')
        # we double the amount of channels every downsampling step up to a max of filters_max
        self.filters_list = [min([self.filters*2**i, self.filters_max])
                             for i in range(self.n_stages)]

        # first let's make the lists for the blocks on both pathes
        # input and output channels
        self.in_channels_list = [self.filters] + self.filters_list[:-1]
        self.out_channels_list = self.filters_list
        # initial strides for downsampling
        self.first_stride_list = [1] + [get_stride(ks) for ks in self.kernel_sizes[:-1]]
        # number of channels we feed forward from the contracting to the expaning path
        self.n_skip_channels = [min([max([1, int(self.factor_skip_conn * f)]), f - 1])
                                for f in self.filters_list]
        # now the upsampling
        self.up_conv_in_list = self.out_channels_list[1:]
        self.up_conv_out_list = [out_ch - skip_ch for out_ch, skip_ch in zip(self.out_channels_list,
                                                                             self.n_skip_channels)]
        # determine how many scales on the upwars path with be connected to
        # a loss function
        if n_pyramid_scales is None:
            self.n_pyramid_scales = max([1, self.n_stages - 1])
        else:
            self.n_pyramid_scales = int(n_pyramid_scales)

        # we first apply one 1x1(x1) convolution to precent information loss as the ReLU is the
        # first module in the conv block
        if self.is_2d:
            self.preprocess = WSConv2d(self.in_channels, self.filters, 1)
        else:
            self.preprocess = WSConv3d(self.in_channels, self.filters, 1)

        # first the downsampling blocks
        self.blocks_down = []
        for in_ch, out_ch, ks, fs, n_bl in zip(self.in_channels_list,
                                               self.out_channels_list,
                                               self.kernel_sizes,
                                               self.first_stride_list,
                                               self.n_blocks):
            block = nfConvStage(in_channels=in_ch,
                                out_channels=out_ch,
                                is_2d=self.is_2d,
                                n_blocks=n_bl,
                                kernel_size=ks,
                                first_stride=fs,
                                conv_params=self.conv_params,
                                nonlin_params=self.nonlin_params,
                                use_bottleneck=self.use_bottleneck,
                                se_reduction=self.se_reduction,
                                alpha=self.alpha,
                                stochdepth_rate=self.stochdepth_rate)
            self.blocks_down.append(block)

        # now the upsampling blocks, note that the number of input channels equals the number of
        # output channels to save a convolution on the skip connections there
        self.blocks_up = []
        for channels, ks in zip(self.out_channels_list[:-1], self.kernel_sizes[:-1]):
            block = nfConvStage(in_channels=channels,
                                out_channels=channels,
                                is_2d=self.is_2d,
                                n_blocks=1,
                                kernel_size=ks,
                                first_stride=1,
                                conv_params=self.conv_params,
                                nonlin_params=self.nonlin_params,
                                use_bottleneck=self.use_bottleneck,
                                se_reduction=self.se_reduction,
                                alpha=self.alpha,
                                stochdepth_rate=self.stochdepth_rate)
            self.blocks_up.append(block)

        # now let's do the upsamplings
        self.upconvs = []
        for in_ch, out_ch, ks in zip(self.up_conv_in_list,
                                     self.up_conv_out_list,
                                     self.kernel_sizes):
            up = UpConv(in_channels=in_ch,
                        out_channels=out_ch,
                        is_2d=is_2d,
                        kernel_size=get_stride(ks),
                        upsampling=self.upsampling,
                        align_corners=self.align_corners)
            self.upconvs.append(up)

        # now the concats:
        self.concats = []
        for in_ch in self.up_conv_out_list:
            if self.use_attention_gates:
                self.concats.append(concat_attention(in_channels=in_ch,
                                                     is_2d=self.is_2d))
            else:
                self.concats.append(concat())

        # last but not least all the logits
        self.all_logits = []
        for in_ch in self.out_channels_list[:self.n_pyramid_scales]:
            self.all_logits.append(Logits(in_channels=in_ch,
                                          out_channels=self.out_channels,
                                          is_2d=self.is_2d,
                                          dropout_rate=self.dropout_rate))

        # now important let's turn everything into a module list
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upconvs = nn.ModuleList(self.upconvs)
        self.concats = nn.ModuleList(self.concats)
        self.all_logits = nn.ModuleList(self.all_logits)

    def forward(self, xb):
        # keep all out tensors from the contracting path
        xb_list = []
        logs_list = []
        # contracting path
        xb = self.preprocess(xb)
        for i in range(self.n_stages):
            xb = self.blocks_down[i](xb)
            # new feature: we only forward half of the channels
            xb_list.append(xb[:, :self.n_skip_channels[i]])

        # expanding path without logits
        for i in range(self.n_stages - 2, self.n_pyramid_scales-1, -1):
            xb = self.upconvs[i](xb)
            xb = self.concats[i](xb, xb_list[i])
            xb = self.blocks_up[i](xb)

        # expanding path with logits
        for i in range(self.n_pyramid_scales - 1, -1, -1):
            xb = self.upconvs[i](xb)
            xb = self.concats[i](xb, xb_list[i])
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i](xb)
            logs_list.append(logs)

        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]


# %%
if __name__ == '__main__':
    gpu = torch.device('cuda:0')
    net = nfUNet(in_channels=1, out_channels=2, kernel_sizes=[(1, 3, 3), (3, 3, 3), 3, 3],
                 is_2d=False,
                 filters=8, factor_skip_conn=0.5, use_bottleneck=False,
                 upsampling='linear').cuda()
    xb = torch.randn((1, 1, 32, 64, 64), device=gpu)
    # xb = torch.randn((3, 1, 512, 512), device=gpu)
    with torch.no_grad():
        yb = net(xb)
    print('Output shapes:')
    for log in yb:
        print(log.shape)