import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch_radon import Radon, RadonFanbeam


def Subho_recon_network():
    device = 'cuda'
    n_angles = 256 #512
    img_size = 512
    det_count = int(np.sqrt(2)*img_size + 0.5)
    radon = Radon(resolution=512,
                  angles=np.linspace(0, np.pi, n_angles, endpoint=False),
                  clip_to_circle=False,
                  det_count=det_count)
    filt_size = 75 #35 working well
    pad = (filt_size-1)//2
    
    class filter_data_space(nn.Module):
        def __init__(self, n_in_channels=1, n_out_channels =1):
            super(filter_data_space, self).__init__()
            
            self.conv1 = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=(1, filt_size), stride=(1, 1), padding=(0, pad), groups = 1,  bias=False)
            self.conv2 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=(1, filt_size), stride=(1, 1), padding=(0, pad), groups = 1,  bias=False)
            self.conv3 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=(1, filt_size), stride=(1, 1), padding=(0, pad), groups = 1,  bias=False)
            self.conv4 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=(1, filt_size), stride=(1, 1), padding=(0, pad), groups = 1,  bias=False)
            
            self.act1 = nn.PReLU(num_parameters=1, init=0.25)
            self.act2 = nn.PReLU(num_parameters=1, init=0.25)
            self.act3 = nn.PReLU(num_parameters=1, init=0.25)
            
        def forward(self, y):
            yf = self.act1(self.conv1(y))
            yf = self.act2(self.conv2(yf))
            yf = self.act3(self.conv3(yf))
            yf = self.conv4(yf)
            return yf
     
     
    #CNN in the image space
    filt_size_image = 5
    pad_image = (filt_size_image-1)//2
    class filter_image_space(nn.Module):
        def __init__(self, n_in_channels=1, n_filters=5, n_out_channels =1):
            super(filter_image_space, self).__init__()
            
            self.conv1 = nn.Conv2d(n_in_channels, n_filters, kernel_size=filt_size_image, stride=1, padding=pad_image, bias=False)
            self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=filt_size_image, stride=1, padding=pad_image, bias=False)
            self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=filt_size_image, stride=1, padding=pad_image, bias=False)
            self.conv4 = nn.Conv2d(n_filters, n_out_channels, kernel_size=filt_size_image, stride=1, padding=pad_image, bias=False)
            
            self.act1 = nn.PReLU(num_parameters=1, init=0.25)
            self.act2 = nn.PReLU(num_parameters=1, init=0.25)
            self.act3 = nn.PReLU(num_parameters=1, init=0.25)
            
        def forward(self, x):
            xf = self.act1(self.conv1(x))
            xf = self.act2(self.conv2(xf))
            xf = self.act3(self.conv3(xf))
            xf = self.conv4(xf)
            return xf
    
    tau = 0.001*torch.ones(1).to(device)
    sigma = 0.001*torch.ones(1).to(device)
    class learned_reconstruction_model(nn.Module):
        def __init__(self, niter=4,tau=tau, sigma=sigma, radon=radon):
            super(learned_reconstruction_model, self).__init__()
            self.radon = radon
            self.niter = niter
            self.filt = nn.ModuleList([filter_data_space().to(device) for i in range(self.niter)])
            self.filt_image = nn.ModuleList([filter_image_space().to(device) for i in range(self.niter)])
            self.tau = nn.Parameter(tau * torch.ones(self.niter).to(device))
            self.sigma = nn.Parameter(sigma * torch.ones(self.niter).to(device))
        def forward(self, y):
            filtered_sinogram = radon.filter_sinogram(y)
            x = radon.backprojection(filtered_sinogram)
            for iteration in range(self.niter):
                res = y - radon.forward(x)
                res_filt = self.filt[iteration](res)
                x += self.tau[iteration]*radon.backprojection(res_filt)
                dx = self.filt_image[iteration](x)
                x = x + self.tau[iteration]*dx #filter the image
            return x
    return learned_reconstruction_model()
    
    