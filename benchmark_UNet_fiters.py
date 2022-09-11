import torch
import torch.nn as nn
import argparse
from contextlib import nullcontext
from time import perf_counter
from ovseg.networks.UNet import UNet


parser = argparse.ArgumentParser()
parser.add_argument("filters", type=int)
args = parser.parse_args()

n_warumup = 25
n_benchmark = 100

nb, nch, nz, nx, ny = 2, 1, 24, 192, 192
params = {'in_channels': 1,
        'out_channels': 2,
        'is_2d': False,
        'filters': args.filters,
        'filters_max': 10*args.filters,
        'conv_params': None,
        'norm': None,
        'norm_params': None,
        'nonlin_params': None,
        'kernel_sizes': [(1, 3, 3), (1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]}

net = UNet(**params).cuda()
xb = torch.zeros((nb, nch, nz, nx, ny), device='cuda')

with torch.cuda.amp.autocast():
    for _ in range(n_warumup):
        out = net(xb)
        l = out[0][:, :1].abs().mean()
        l.backward()
    
    net.zero_grad()
    
    st = perf_counter()
    with torch.no_grad():
        for _ in range(n_benchmark):
            out = net(xb)
    et = perf_counter()
    
    print('{:.3e}'.format((et-st)/n_benchmark))
    st = perf_counter()
    for _ in range(n_benchmark):
        out = net(xb)
        l = out[0][:, :1].abs().mean()
        l.backward()
        net.zero_grad()
    et = perf_counter()
    print('{:.3e}'.format((et-st)/n_benchmark))
