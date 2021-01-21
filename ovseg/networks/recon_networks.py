import numpy as np
import torch
import torch.nn as nn
from torch_radon import Radon


def get_operator(n_angles=256, det_count=724):
    radon = Radon(resolution=512,
                  angles=np.linspace(0, np.pi, n_angles, endpoint=False),
                  clip_to_circle=False,
                  det_count=det_count)
    return radon


class filter_data_space(nn.Module):
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

    def forward(self, x):
        xf = self.act1(self.conv1(x))
        xf = self.act2(self.conv2(xf))
        xf = self.act3(self.conv3(xf))
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
        self.filt = nn.ModuleList([filter_data_space().to(device) for i in range(self.niter)])
        self.filt_image = nn.ModuleList([filter_image_space().to(device)
                                         for i in range(self.niter)])
        self.tau = nn.Parameter(tau * torch.ones(self.niter).to(device))
        self.sigma = nn.Parameter(sigma * torch.ones(self.niter).to(device))

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
        return x


class proximal_convs(nn.Module):
    def __init__(self, n_in, n_hid=32, n_out=5):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, n_hid, 3, padding=1)
        self.conv2 = nn.Conv2d(n_hid, n_hid, 3, padding=1)
        self.conv3 = nn.Conv2d(n_hid, n_out, 3, padding=1)
        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, inpt):
        return self.conv3(self.act2(self.conv2(self.act1(self.conv1(inpt)))))


class proximal_dual(nn.Module):
    def __init__(self, radon, sigma_init=0.001):
        super().__init__()
        self.prox_conv = proximal_convs(7)
        self.radon = radon
        self.sigma = nn.Parameter(torch.ones(1)*sigma_init)

    def forward(self, h, f, g):
        Kf = self.radon.forward(f[:, 1:2])
        return h + self.prox_conv(torch.cat([h, Kf, g], 1))


class proximal_primal(nn.Module):
    def __init__(self, radon, tau_init=0.001):
        super().__init__()
        self.prox_conv = proximal_convs(6)
        self.radon = radon
        self.tau = nn.Parameter(torch.ones(1)*tau_init)

    def forward(self, h, f):
        Kadjh = self.radon.backprojection(h[:, 0:1])
        return f + self.prox_conv(torch.cat([f, Kadjh], 1))


class proximal_update(nn.Module):
    def __init__(self, radon, sigma_init=0.001, tau_init=0.001):
        super().__init__()
        self.prox_primal = proximal_primal(radon, tau_init)
        self.prox_dual = proximal_dual(radon, sigma_init)

    def forward(self, h, f, g):
        h_new = self.prox_dual(h, f, g)
        f_new = self.prox_primal(h_new, f)
        return h_new, f_new


class learned_primal_dual(nn.Module):
    def __init__(self, radon, n_inter=10):
        super().__init__()
        self.prox_updates = nn.ModuleList([proximal_update(radon) for _ in range(n_inter)])
        self.radon = radon

    def forward(self, g):
        filtered_sinogram = self.radon.filter_sinogram(g)
        f = self.radon.backprojection(filtered_sinogram)
        f = torch.cat([f for _ in range(5)], 1)
        size = (g.shape[0], 5, len(self.radon.angles), self.radon.det_count)
        h = torch.zeros(size, device=g.device)

        for prox in self.prox_updates:
            h, f = prox(h, f, g)

        return f[:, :1]
