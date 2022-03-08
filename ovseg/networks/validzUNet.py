import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

m_down_default = [3, 3, 3, 1, 1, 1 ,0]
n_down_default = [0, 0, 0, 2, 2, 2, 4]
m_up_default = [4, 4, 4, 3, 3, 3, 3]

# %%
class basic_block_down(nn.Module):

    def __init__(self, m, n, downsample, ch_in, ch_out):

        super().__init__()
        self.m = m
        self.n = n
        self.downsample = downsample
        self.ch_in = ch_in
        self.ch_out = ch_out

            
        self.skip = nn.Conv3d(self.ch_in, self.ch_out, 1)
        nn.init.kaiming_normal_(self.skip.weight)
        
        in_channels = [self.ch_in] + [self.ch_out for _ in range(self.m+self.n-1)]
        strides = [1 for _ in range(self.m + self.n)]
        if self.downsample:
            pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))
            self.skip = nn.Sequential(pool, self.skip)
            strides[0] = (1, 2, 2)

        modules = [nn.ReLU(inplace=False)]
        # first 2d convolutions
        for in_ch, stride in zip(in_channels[:self.m], strides[:self.m]):
            modules.append(nn.Conv3d(in_ch,
                                     self.ch_out,
                                     (1, 3, 3),
                                     stride=stride,
                                     padding=(0, 1, 1)))
            modules.append(nn.ReLU(inplace=True))
        # now 3d convolutions
        for in_ch, stride in zip(in_channels[self.m:], strides[self.m:]):
            modules.append(nn.Conv3d(in_ch,
                                     self.ch_out,
                                     (3, 3, 3),
                                     stride=stride,
                                     padding=(0, 1, 1)))
            modules.append(nn.ReLU(inplace=True))

        # remove last ReLU
        modules = modules[:-1]
        self.mlp = nn.Sequential(*modules)

    def forward(self, xb):
        skip = self.skip(xb)
        if self.n > 0:
            skip = skip[:, :, self.n:-self.n]
        return skip + self.mlp(xb)


class basic_block_up(nn.Module):

    def __init__(self, m, ch_in, ch_out):

        super().__init__()
        self.m = m
        self.ch_in = ch_in
        self.ch_out = ch_out

            
        self.skip = nn.Conv3d(2*self.ch_in, self.ch_out, 1)
        nn.init.kaiming_normal_(self.skip.weight)
        
        in_channels = [2*self.ch_in] + [self.ch_out for _ in range(self.m-1)]

        modules = [nn.ReLU(inplace=False)]
        # first 2d convolutions
        for in_ch in in_channels:
            modules.append(nn.Conv3d(in_ch,
                                     self.ch_out,
                                     (1, 3, 3),
                                     padding=(0, 1, 1)))
            modules.append(nn.ReLU(inplace=True))

        # remove last ReLU
        modules = modules[:-1]
        # stack all modules
        self.mlp = nn.Sequential(*modules)

    def forward(self, xb, xb_up=None):
        if xb_up is not None:
            k = int(xb.shape[2] - xb_up.shape[2]) // 2
            if k > 0:
                xb = torch.cat([xb[:, :, k:-k], xb_up], 1)
        return self.skip(xb) + self.mlp(xb)

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                                       (1, 2, 2), stride=(1, 2, 2),
                                       bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, xb):
        return self.conv(xb)


class validzUNet(nn.Module):

    def __init__(self, in_ch=1, out_ch=2, filt=32, m_down=None, n_down=None, m_up=None):
        super().__init__()
        self.filt=filt
        self.m_down=m_down if m_down is not None else m_down_default
        self.n_down=n_down if n_down is not None else n_down_default
        self.m_up=m_up if m_up is not None else m_up_default
        self.is_2d=False
        self.in_channels = in_ch
        self.out_channels = out_ch

        assert len(self.m_down) == len(self.n_down)
        assert len(self.m_down) == len(self.m_up)

        self.n_z_convs = np.sum(self.n_down)
        # filters in each block, we double only every other
        self.filters_list = self.filt * 2 ** (np.arange(len(self.m_down)) // 2)

        # init downsampling blocks
        self.blocks_down = [basic_block_down(self.m_down[0],
                                             self.n_down[0], 
                                             False,
                                             self.in_channels,
                                             self.filt)]
        for i in range(1, len(self.m_down)):
            self.blocks_down.append(basic_block_down(self.m_down[i],
                                                     self.n_down[i],
                                                     True,
                                                     self.filters_list[i-1],
                                                     self.filters_list[i]))

        # now init upsampling blocks
        self.blocks_up = []
        self.upsamplings = []
        for i in range(len(self.m_up)-1):
            self.blocks_up.append(basic_block_up(self.m_up[i],
                                                 self.filters_list[i],
                                                 self.filters_list[i]))
            self.upsamplings.append(UpConv(self.filters_list[i+1], self.filters_list[i]))

        self.blocks_up.append(basic_block_up(self.m_up[-1],
                                             self.filters_list[-1]//2,
                                             self.filters_list[-1]))

        # don't forget to turn everything into module lists
        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.blocks_up = nn.ModuleList(self.blocks_up)
        self.upsamplings = nn.ModuleList(self.upsamplings)

        self.logits = nn.Conv3d(self.filt, self.out_channels, 1, bias=False)

    def forward(self, xb):

        xb_skip_list = []
        for block in self.blocks_down:
            xb = block(xb)
            xb_skip_list.append(xb)

        # the lowest block of the upsampling path
        xb = self.blocks_up[-1](xb)
        for i in range(len(self.upsamplings)-1, -1, -1):
            xb_up = self.upsamplings[i](xb)
            xb_skip = xb_skip_list[i]
            xb = self.blocks_up[i](xb_skip, xb_up)
        
        return self.logits(xb)


# %%
if __name__ == '__main__':

    xb_train = torch.randn((1, 1, 21, 512, 512)).cuda()
    xb_val = torch.randn((1, 1, 31, 512, 512)).cuda()
    net = validzUNet(filt=8).cuda()

    print('train')
    yb_train = net(xb_train)
    print(yb_train.shape)
    print('val')
    with torch.no_grad():
        yb_val = net(xb_val)

    print(yb_val.shape)