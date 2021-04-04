import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import perf_counter
from ovseg.networks.blocks import nfConvResStage, nfConvBlock, get_stride, WSConv2d, WSConv3d


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
                 is_2d=False, filters=32, filters_max=320, n_pyramid_scales=None,
                 conv_params=None, nonlin_params=None, use_attention_gates=False, upsampling='conv',
                 align_corners=True, factor_skip_conn=0.5, is_inference_network=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.is_2d = is_2d
        self.n_stages = len(kernel_sizes)
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.nonlin_params = nonlin_params
        self.use_attention_gates = use_attention_gates
        self.upsampling = upsampling
        self.align_corners = align_corners
        self.factor_skip_conn = factor_skip_conn
        self.is_inference_network = is_inference_network
        if self.factor_skip_conn >= 1 or self.factor_skip_conn <= 0:
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
            self.n_pyramid_scales = max([1, self.n_stages - 2])
        else:
            self.n_pyramid_scales = int(n_pyramid_scales)

        # we first apply one 1x1(x1) convolution to precent information loss as the ReLU is the
        # first module in the conv block
        if self.is_2d:
            self.preprocess = nn.Conv2d(self.in_channels, self.filters, 1)
        else:
            self.preprocess = nn.Conv3d(self.in_channels, self.filters, 1)

        nn.init.kaiming_normal_(self.preprocess.weight, nonlinearity='relu')
        nn.init.zeros_(self.preprocess.bias)

        # first the downsampling blocks
        self.blocks_down = []
        for in_ch, out_ch, ks, fs in zip(self.in_channels_list,
                                         self.out_channels_list,
                                         self.kernel_sizes,
                                         self.first_stride_list):
            block = nfConvBlock(in_channels=in_ch,
                                out_channels=out_ch,
                                is_2d=self.is_2d,
                                kernel_size=ks,
                                first_stride=fs,
                                conv_params=self.conv_params,
                                nonlin_params=self.nonlin_params,
                                is_inference_block=self.is_inference_network)
            self.blocks_down.append(block)

        # now the upsampling blocks, note that the number of input channels equals the number of
        # output channels to save a convolution on the skip connections there
        self.blocks_up = []
        for channels, ks in zip(self.out_channels_list[:-1], self.kernel_sizes[:-1]):
            block = nfConvBlock(in_channels=channels,
                                out_channels=channels,
                                is_2d=self.is_2d,
                                kernel_size=ks,
                                first_stride=1,
                                conv_params=self.conv_params,
                                nonlin_params=self.nonlin_params,
                                is_inference_block=self.is_inference_network)
            # we don't use stoch depth on the upsampling pathes
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

# %% normalization free U-Nets with residual connections
class nfUResNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes,
                 is_2d=False, n_blocks=None, filters=32, filters_max=320, n_pyramid_scales=None,
                 conv_params=None, nonlin_params=None, use_bottleneck=False, bottleneck_ratio=2,
                 se_reduction=4, use_attention_gates=False, alpha=0.2, stochdepth_rate=0,
                 dropout_rate=0, upsampling='conv', align_corners=True, factor_skip_conn=0.5):
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
        self.bottleneck_ratio = bottleneck_ratio
        self.se_reduction = se_reduction
        self.use_attention_gates = use_attention_gates
        self.alpha = alpha
        self.stochdepth_rate = stochdepth_rate
        self.dropout_rate = dropout_rate
        self.upsampling = upsampling
        self.align_corners = align_corners
        self.factor_skip_conn = factor_skip_conn
        if self.factor_skip_conn >= 1 or self.factor_skip_conn <= 0:
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
            self.preprocess = nn.Conv2d(self.in_channels, self.filters, 1)
        else:
            self.preprocess = nn.Conv3d(self.in_channels, self.filters, 1)

        nn.init.kaiming_normal_(self.preprocess.weight, nonlinearity='relu')
        nn.init.zeros_(self.preprocess.bias)

        # first the downsampling blocks
        self.blocks_down = []
        for in_ch, out_ch, ks, fs, n_bl in zip(self.in_channels_list,
                                               self.out_channels_list,
                                               self.kernel_sizes,
                                               self.first_stride_list,
                                               self.n_blocks):
            block = nfConvResStage(in_channels=in_ch,
                                   out_channels=out_ch,
                                   is_2d=self.is_2d,
                                   n_blocks=n_bl,
                                   kernel_size=ks,
                                   first_stride=fs,
                                   conv_params=self.conv_params,
                                   nonlin_params=self.nonlin_params,
                                   use_bottleneck=self.use_bottleneck,
                                   bottleneck_ratio=self.bottleneck_ratio,
                                   se_reduction=self.se_reduction,
                                   alpha=self.alpha,
                                   stochdepth_rate=self.stochdepth_rate)
            self.blocks_down.append(block)

        # now the upsampling blocks, note that the number of input channels equals the number of
        # output channels to save a convolution on the skip connections there
        self.blocks_up = []
        for channels, ks in zip(self.out_channels_list[:-1], self.kernel_sizes[:-1]):
            block = nfConvResStage(in_channels=channels,
                                   out_channels=channels,
                                   is_2d=self.is_2d,
                                   n_blocks=1,
                                   kernel_size=ks,
                                   first_stride=1,
                                   conv_params=self.conv_params,
                                   nonlin_params=self.nonlin_params,
                                   use_bottleneck=self.use_bottleneck,
                                   bottleneck_ratio=self.bottleneck_ratio,
                                   se_reduction=self.se_reduction,
                                   alpha=self.alpha,
                                   stochdepth_rate=0.)
            # we don't use stoch depth on the upsampling pathes
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


class nfUResNet_benchmark(nfUResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zero_perf_times()

    def _zero_perf_times(self):
        self.perf_time_down = [0 for _ in range(self.n_stages)]
        self.perf_time_upsampling = [0 for _ in range(self.n_stages-1)]
        self.perf_time_concat = [0 for _ in range(self.n_stages-1)]
        self.perf_time_up = [0 for _ in range(self.n_stages-1)]

    def _print_perf_times(self):
        total_time = np.sum(self.perf_time_down) + np.sum(self.perf_time_up) + \
            np.sum(self.perf_time_upsampling) + np.sum(self.perf_time_concat)
        perc_down = 100 * np.array(self.perf_time_down) / total_time
        perc_upsampling = 100 * np.array(self.perf_time_upsampling) / total_time
        perc_concat = 100 * np.array(self.perf_time_concat) / total_time
        perc_up = 100 * np.array(self.perf_time_up) / total_time
        print(*perc_down)
        print(*perc_upsampling)
        print(*perc_concat)
        print(*perc_up)

    def forward(self, xb):
        # keep all out tensors from the contracting path
        xb_list = []
        logs_list = []
        # contracting path
        xb = self.preprocess(xb)
        for i in range(self.n_stages):
            t = perf_counter()
            xb = self.blocks_down[i](xb)
            # new feature: we only forward half of the channels
            xb_list.append(xb[:, :self.n_skip_channels[i]])
            torch.cuda.synchronize()
            self.perf_time_down[i] += perf_counter() - t

        # expanding path without logits
        for i in range(self.n_stages - 2, self.n_pyramid_scales-1, -1):
            t = perf_counter()
            xb = self.upconvs[i](xb)
            torch.cuda.synchronize()
            self.perf_time_upsampling[i] += perf_counter() - t
            t = perf_counter()
            xb = self.concats[i](xb, xb_list[i])
            torch.cuda.synchronize()
            self.perf_time_concat[i] += perf_counter() - t
            t = perf_counter()
            xb = self.blocks_up[i](xb)
            torch.cuda.synchronize()
            self.perf_time_up[i] += perf_counter() - t

        # expanding path with logits
        for i in range(self.n_pyramid_scales - 1, -1, -1):
            t = perf_counter()
            xb = self.upconvs[i](xb)
            torch.cuda.synchronize()
            self.perf_time_upsampling[i] += perf_counter() - t
            t = perf_counter()
            xb = self.concats[i](xb, xb_list[i])
            torch.cuda.synchronize()
            self.perf_time_concat[i] += perf_counter() - t
            t = perf_counter()
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i](xb)
            logs_list.append(logs)
            torch.cuda.synchronize()
            self.perf_time_up[i] += perf_counter() - t

        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]


# %%
if __name__ == '__main__':
    gpu = torch.device('cuda:0')
    net = nfUResNet(in_channels=1, out_channels=2, kernel_sizes=[(1, 3, 3), (3, 3, 3), 3, 3],
                    is_2d=False,
                    filters=8, factor_skip_conn=0.5, use_bottleneck=True,
                    upsampling='linear').cuda()
    xb = torch.randn((1, 1, 32, 64, 64), device=gpu)
    # xb = torch.randn((3, 1, 512, 512), device=gpu)
    with torch.no_grad():
        yb = net(xb)
    print('Output shapes:')
    for log in yb:
        print(log.shape)