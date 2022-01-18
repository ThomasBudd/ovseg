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
    
    def __init__(self, n_channels, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(n_channels, **kwargs)
    
    def forward(self, xb):
        
        nb, nc, nz, nx, ny = xb.shape
        
        # move channels to last dimension
        xb = xb.permute((0, 2, 3, 4, 1))
        xb = self.norm(xb)
        # undo
        xb = xb.permute((0, 4, 1, 2, 3))
        
        return xb
        