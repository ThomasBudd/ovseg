import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

GAMMA_RELU = 1.7128585504496627  # =np.sqrt(2 / (1 - 1/np.pi))

class scaledReLU(nn.ReLU):
    # TODO integrate the scale here so that we can put the alpha and beta in here
    def forward(self, input):
        return GAMMA_RELU * super().forward(input)

# %%
class ConvNormNonlinBlock(nn.Module):

    def __init__(self, in_channels, out_channels, is_2d, kernel_size=3,
                 first_stride=1, conv_params=None, norm=None, norm_params=None,
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

    def forward(self, xb):
        xb = self.conv1(xb)
        xb = self.norm1(xb)
        xb = self.nonlin1(xb)
        xb = self.conv2(xb)
        xb = self.norm2(xb)
        xb = self.nonlin2(xb)
        return xb
    
# %% normalization free Blocks
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
            self.logits = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.logits = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        nn.init.kaiming_normal_(self.logits.weight)

    def forward(self, xb):
        return self.logits(xb)


# %%

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes,
                 is_2d, filters=32, filters_max=384, n_pyramid_scales=None,
                 conv_params=None, norm=None, norm_params=None, nonlin_params=None,
                 kernel_sizes_up=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.is_2d = is_2d
        self.n_stages = len(kernel_sizes)
        self.filters = filters
        self.filters_max = filters_max
        self.conv_params = conv_params
        self.norm = norm
        self.norm_params = norm_params
        self.nonlin_params = nonlin_params
        self.kernel_sizes_up = kernel_sizes_up if kernel_sizes_up is not None else kernel_sizes[:-1]
        # we double the amount of channels every downsampling step
        # up to a max of filters_max
        self.filters_list = [min([self.filters*2**i, self.filters_max])
                             for i in range(self.n_stages)]

        # first let's make the lists for the blocks on both pathes
        self.in_channels_down_list = [self.in_channels] + self.filters_list[:-1]
        self.out_channels_down_list = self.filters_list
        self.first_stride_list = [1] + [get_stride(ks) for ks in self.kernel_sizes[:-1]]
        self.in_channels_up_list = [2 * n_ch for n_ch in self.out_channels_down_list[:-1]]
        self.out_channels_up_list = self.out_channels_down_list[:-1]

        # now the upconvolutions
        self.up_conv_in_list = self.out_channels_down_list[1:]
        self.up_conv_out_list = self.out_channels_down_list[:-1]

        # determine how many scales on the upwars path with be connected to
        # a loss function
        if n_pyramid_scales is None:
            self.n_pyramid_scales = max([1, self.n_stages - 2])
        else:
            self.n_pyramid_scales = int(n_pyramid_scales)

        # now all the logits
        self.logits_in_list = self.out_channels_up_list[:self.n_pyramid_scales]

        # blocks on the contracting path
        self.blocks_down = []
        for in_channels, out_channels, kernel_size, first_stride in zip(self.in_channels_down_list,
                                                                        self.out_channels_down_list,
                                                                        self.kernel_sizes,
                                                                        self.first_stride_list):
            # block = ConvNormNonlinBlock(in_channels=in_channels,
            #                             out_channels=out_channels,
            #                             is_2d=self.is_2d,
            #                             kernel_size=kernel_size,
            #                             first_stride=first_stride,
            #                             conv_params=self.conv_params,
            #                             norm=self.norm,
            #                             norm_params=self.norm_params,
            #                             nonlin_params=self.nonlin_params)
            block = nfConvBlock(in_channels=in_channels,
                                out_channels=out_channels,
                                is_2d=self.is_2d,
                                kernel_size=kernel_size,
                                first_stride=first_stride)
            self.blocks_down.append(block)

        # blocks on the upsampling path
        self.blocks_up = []
        for in_channels, out_channels, kernel_size in zip(self.in_channels_up_list,
                                                          self.out_channels_up_list,
                                                          self.kernel_sizes_up):
            # block = ConvNormNonlinBlock(in_channels=in_channels,
            #                             out_channels=out_channels,
            #                             is_2d=self.is_2d,
            #                             kernel_size=kernel_size,
            #                             conv_params=self.conv_params,
            #                             norm=self.norm,
            #                             norm_params=self.norm_params,
            #                             nonlin_params=self.nonlin_params)
            block = nfConvBlock(in_channels=in_channels,
                                out_channels=out_channels,
                                is_2d=self.is_2d,
                                kernel_size=kernel_size)
            self.blocks_up.append(block)

        # upsaplings
        self.upconvs = []
        for in_channels, out_channels, kernel_size in zip(self.up_conv_in_list,
                                                          self.up_conv_out_list,
                                                          self.kernel_sizes):
            self.upconvs.append(UpConv(in_channels=in_channels,
                                       out_channels=out_channels,
                                       is_2d=self.is_2d,
                                       kernel_size=get_stride(kernel_size)))

        # logits
        self.all_logits = []
        for in_channels in self.logits_in_list:
            self.all_logits.append(Logits(in_channels=in_channels,
                                          out_channels=self.out_channels,
                                          is_2d=self.is_2d))

        # now important let's turn everything into a module list
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upconvs = nn.ModuleList(self.upconvs)
        self.all_logits = nn.ModuleList(self.all_logits)
        
        self.stats_decoder = np.zeros((6, 2))
        self.stats_encoder = np.zeros((5, 2))

    def forward(self, xb):
        # keep all out tensors from the contracting path
        xb_list = []
        logs_list = []
        # contracting path
        for i, block in enumerate(self.blocks_down):
            xb = block(xb)
            xb_list.append(xb)
            self.stats_decoder[i, 0] += (xb.mean(dim=(0, 2, 3, 4)) ** 2).mean().cpu().detach().numpy()
            self.stats_decoder[i, 1] += (xb.var(dim=(0, 2, 3, 4))).mean().cpu().detach().numpy()

        # expanding path without logits
        for i in range(self.n_stages - 2, self.n_pyramid_scales-1, -1):
            xb = self.upconvs[i](xb)
            xb = torch.cat([xb, xb_list[i]], 1)
            del xb_list[i]
            xb = self.blocks_up[i](xb)
            self.stats_encoder[i, 0] += (xb.mean(dim=(0, 2, 3, 4)) ** 2).mean().cpu().detach().numpy()
            self.stats_encoder[i, 1] += (xb.var(dim=(0, 2, 3, 4)) ** 2).mean().cpu().detach().numpy()

        # expanding path with logits
        for i in range(self.n_pyramid_scales - 1, -1, -1):
            xb = self.upconvs[i](xb)
            xb = torch.cat([xb, xb_list[i]], 1)
            del xb_list[i]
            xb = self.blocks_up[i](xb)
            logs = self.all_logits[i](xb)
            logs_list.append(logs)
            self.stats_encoder[i][0] += (xb.mean(dim=(0, 2, 3, 4)) ** 2).mean().cpu().detach().numpy()
            self.stats_encoder[i][1] += (xb.var(dim=(0, 2, 3, 4)) ** 2).mean().cpu().detach().numpy()

        # as we iterate from bottom to top we have to flip the logits list
        return logs_list[::-1]

# %%
net = UNet(1, 2, [(1, 3, 3), (1, 3, 3), 3, 3, 3, 3], is_2d=False,
           nonlin_params = {'negative_slope': 0.0, 'inplace': True}).cuda()

with torch.no_grad():
    for _ in range(25):
        net(torch.randn(2, 1, 48, 192, 192).cuda())

net.stats_decoder /= 25
net.stats_encoder /= 25
print(net.stats_decoder)
print(net.stats_encoder)