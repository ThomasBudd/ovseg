import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


in_channels, hid_channels, out_channels = 1, 32, 32
kernel_size, padding, first_stride = 3, 1, 1

conv_params = {'bias':False}
norm_params = {'affine': True}

p_dropout = 0.1

nonlin_params = {'negative_slope': 0.01, 'inplace': True}

conv_fctn = nn.Conv3d
norm_fctn = nn.InstanceNorm3d
drop_fctn = nn.Dropout3d

layers = []
        
conv1 = conv_fctn(in_channels, hid_channels,
                       kernel_size, padding=padding,
                       stride=first_stride, **conv_params)

nn.init.kaiming_normal_(conv1.weight)
layers.append(conv1)
layers.append(norm_fctn(hid_channels, **norm_params))

if p_dropout > 0:
    layers.append(drop_fctn(p_dropout))
    
layers.append(nn.LeakyReLU(**nonlin_params))

# now again
conv2 = conv_fctn(hid_channels, out_channels,
                 kernel_size, padding=padding,
                 **conv_params)
nn.init.kaiming_normal_(conv2.weight)
layers.append(conv2)
layers.append(norm_fctn(out_channels, **norm_params))

if p_dropout > 0:
    layers.append(drop_fctn(p_dropout))
    
layers.append(nn.LeakyReLU(**nonlin_params))

# turn into a sequential module
# modules = nn.ModuleList(layers)
modules = nn.Sequential(*layers).cuda()

xb = torch.randn((1, 1, 32, 128, 128), device='cuda')

out = modules(xb)

print(out)
