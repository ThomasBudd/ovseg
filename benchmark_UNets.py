import torch
import torch.nn as nn
import argparse
from contextlib import nullcontext
from time import perf_counter
from ovseg.networks.UNet import UNet


parser = argparse.ArgumentParser()
parser.add_argument("net", type=int)
parser.add_argument('--fp32', default=False, action='store_true')
args = parser.parse_args()

n_warumup = 25
n_benchmark = 100

nb, nch, nz, nx, ny = 1, 1, 32, 256, 256

if args.net == 0:
    
    params = {'in_channels': 1,
             'out_channels': 2,
             'is_2d': False,
             'filters': 32,
             'filters_max': 320,
             'conv_params': None,
             'norm': None,
             'norm_params': None,
             'nonlin_params': None,
             'kernel_sizes': [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]}

elif args.net == 1:
    
    params = {'in_channels': 1,
             'out_channels': 2,
             'is_2d': False,
             'filters': 64,
             'filters_max': 320,
             'conv_params': None,
             'norm': None,
             'norm_params': None,
             'nonlin_params': None,
             'kernel_sizes': [(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
             'stem_kernel_size': [1, 2, 2]}

elif args.net == 2:
    
   
    params = {'in_channels': 1,
             'out_channels': 2,
             'is_2d': False,
             'filters': 32,
             'filters_max': 320,
             'conv_params': None,
             'norm': None,
             'norm_params': None,
             'nonlin_params': None,
             'kernel_sizes': [(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
             'stem_kernel_size': [1, 2, 2]}

else:
    raise NotImplementedError(f'Got unkown type of net {args.net}')


net = UNet(**params).cuda()
xb = torch.zeros((nb, nch, nz, nx, ny), device='cuda')

context = nullcontext() if args.fp32 else torch.cuda.amp.autocast()

with context:
    for _ in range(n_warumup):
        out = net(xb)
        l = out.abs().mean()
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
