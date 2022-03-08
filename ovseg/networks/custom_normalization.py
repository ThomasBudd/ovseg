import torch
import torch.nn as nn
import torch.nn.functional as F

class no_z_InstNorm(nn.Module):

    def __init__(self, n_channels, **kwargs):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channels, **kwargs)
    
    def forward(self, xb):
        
        nb, nc, nz, nx, ny = xb.shape
        
        # put the z slices in the batch dimension
        xb = xb.permute((0, 2, 1, 3, 4)).reshape((nb*nz, nc, nx, ny))
        xb = self.norm(xb)
        # undo
        xb = xb.reshape((nb, nz, nc, nx, ny)).permute((0, 2, 1, 3, 4))
        
        return xb

class my_LayerNorm(nn.Module):
    
    def __init__(self, n_channels, affine=True, eps=1e-5):
        super().__init__()
        
        self.n_channels = n_channels
        self.affine = affine
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones((1, self.n_channels, 1, 1, 1)))
        
        if self.affine:
            self.beta = nn.Parameter(torch.zeros((1, self.n_channels, 1, 1, 1)))
            
    
    def forward(self, xb):
        
        # normalize
        xb = (xb - torch.mean(xb, 1, keepdim=True))/(torch.std(xb, 1, unbiased=False, keepdim=True) + self.eps)
        # affine trafo
        xb = xb * self.gamma
        
        if self.affine:
            xb = xb + self.beta
        
        return xb
        