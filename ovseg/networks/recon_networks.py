import numpy as np
import torch
import torch.nn as nn
from torch_radon import Radon
from ovseg.networks.UNet import UNet


def get_operator(n_angles=256, det_count=724):
    radon = Radon(resolution=512,
                  angles=np.linspace(0, np.pi, n_angles, endpoint=False),
                  clip_to_circle=False,
                  det_count=det_count)
    return radon


class filter_data_space_fbp_conv(nn.Module):
    def __init__(self, n_in_channels=1, n_out_channels=1, filt_size=75,):
        super().__init__()
        pad = (filt_size-1)//2
        self.conv1 = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=(1, filt_size),
                               stride=(1, 1), padding=(0, pad), groups=1, bias=False)
        self.conv2 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=(1, filt_size),
                               stride=(1, 1), padding=(0, pad), groups=1, bias=False)
        self.conv3 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=(1, filt_size),
                               stride=(1, 1), padding=(0, pad), groups=1, bias=False)
        self.conv4 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=(1, filt_size),
                               stride=(1, 1), padding=(0, pad), groups=1, bias=False)

        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        self.act3 = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, y):
        yf = self.act1(self.conv1(y))
        yf = self.act2(self.conv2(yf))
        yf = self.act3(self.conv3(yf))
        yf = self.conv4(yf)
        return yf

class filter_data_space(nn.Module):
    def __init__(self, n_in_channels=1, n_out_channels=1, n_hid_channels=5,
                 filt_size=5):
        super().__init__()
        pad = (filt_size-1)//2
        self.conv1 = nn.Conv2d(n_in_channels, n_hid_channels, kernel_size=filt_size,
                               padding=pad, bias=False)
        self.conv2 = nn.Conv2d(n_hid_channels, n_hid_channels, kernel_size=filt_size,
                               padding=pad, bias=False)
        self.conv3 = nn.Conv2d(n_hid_channels, n_hid_channels, kernel_size=filt_size,
                               padding=pad, bias=False)
        self.conv4 = nn.Conv2d(n_hid_channels, n_out_channels, kernel_size=filt_size,
                               padding=pad, bias=False)

        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        self.act3 = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, y):
        yf = self.act1(self.conv1(y))
        yf = self.act2(self.conv2(yf))
        yf = self.act3(self.conv3(yf))
        yf = self.conv4(yf)
        return yf


class filter_image_space(nn.Module):
    def __init__(self, n_in_channels=1, n_filters=5, n_out_channels=1, filt_size_image=5):
        super(filter_image_space, self).__init__()
        pad_image = (filt_size_image-1)//2
        self.conv1 = nn.Conv2d(n_in_channels, n_filters, kernel_size=filt_size_image, stride=1,
                               padding=pad_image, bias=False)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=filt_size_image, stride=1,
                               padding=pad_image, bias=False)
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=filt_size_image, stride=1,
                               padding=pad_image, bias=False)
        self.conv4 = nn.Conv2d(n_filters, n_out_channels, kernel_size=filt_size_image, stride=1,
                               padding=pad_image, bias=False)

        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        self.act3 = nn.PReLU(num_parameters=1, init=0.25)

        self.norm1 = nn.BatchNorm2d(n_filters)
        self.norm2 = nn.BatchNorm2d(n_filters)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        xf = self.act1(self.norm1(self.conv1(x)))
        xf = self.act2(self.norm2(self.conv2(xf)))
        xf = self.act3(self.norm3(self.conv3(xf)))
        xf = self.conv4(xf)
        return xf


class reconstruction_network_fbp_convs(nn.Module):
    def __init__(self, radon, niter=4, denoise_filters=5, denoise_depth=3):

        # first setup everythin for the inversion
        self.radon = radon
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tau = 0.001*torch.ones(1).to(device)
        sigma = 0.001*torch.ones(1).to(device)
        super().__init__()
        self.radon = radon
        self.niter = niter
        self.filt = nn.ModuleList([filter_data_space_fbp_conv().to(device) for i in range(self.niter)])
        self.filt_image = nn.ModuleList([filter_image_space().to(device)
                                         for i in range(self.niter)])
        self.tau = nn.Parameter(tau * torch.ones(self.niter).to(device))
        self.sigma = nn.Parameter(sigma * torch.ones(self.niter).to(device))

        # now the denoising UNet
        self.UNet = UNet(in_channels=1, out_channels=1, kernel_sizes=denoise_depth * [3],
                         is_2d=True, filters=denoise_filters, n_pyramid_scales=1)

    def forward(self, y):
        filtered_sinogram = self.radon.filter_sinogram(y)
        x = self.radon.backprojection(filtered_sinogram)
        for iteration in range(self.niter):
            res = y - self.radon.forward(x)
            res_filt = self.filt[iteration](res)
            x += self.tau[iteration]*self.radon.backprojection(res_filt)
            dx = self.filt_image[iteration](x)
            x = x + self.tau[iteration]*dx
        # the UNet is returning a list of predictions on the different scales.
        # that's why we use the [0]
        return self.UNet(x)[0]


class reconstruction_network_fbp_ops(nn.Module):
    def __init__(self, radon, niter=4, denoise_filters=5, denoise_depth=3):

        # first setup everythin for the inversion
        self.radon = radon
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tau = 0.001*torch.ones(1).to(device)
        sigma = 0.001*torch.ones(1).to(device)
        super().__init__()
        self.radon = radon
        self.niter = niter
        self.filt = nn.ModuleList([filter_data_space().to(device) for i in range(self.niter)])
        self.filt_image = nn.ModuleList([filter_image_space().to(device)
                                         for i in range(self.niter)])
        self.tau = nn.Parameter(tau * torch.ones(self.niter).to(device))
        self.sigma = nn.Parameter(sigma * torch.ones(self.niter).to(device))

        # now the denoising UNet
        self.UNet = UNet(in_channels=1, out_channels=1, kernel_sizes=denoise_depth * [3],
                         is_2d=True, filters=denoise_filters, n_pyramid_scales=1)

    def fbp(self, y):
        y_filt = self.radon.filter_sinogram(y, 'ramp')
        return self.radon.backprojection(y_filt)

    def forward(self, y):
        x = self.fbp(y)
        for iteration in range(self.niter):
            res = y - self.radon.forward(x)
            res_filt = self.filt[iteration](res)
            
            x += self.tau[iteration]*self.fbp(res_filt)
            dx = self.filt_image[iteration](x)
            x = x + self.tau[iteration]*dx
        # the UNet is returning a list of predictions on the different scales.
        # that's why we use the [0]
        return self.UNet(x)[0]
