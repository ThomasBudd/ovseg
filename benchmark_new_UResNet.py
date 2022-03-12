from ovseg.networks.resUNet import UNetResEncoderV2

import torch
import torch.nn as nn
import argparse
from contextlib import nullcontext
from time import perf_counter


parser = argparse.ArgumentParser()
parser.add_argument("bs", type=int)
parser.add_argument("--more_filters", default=False, action='store_true')
parser.add_argument("--blocks", nargs='+', default=None)
parser.add_argument('--fp32', default=False, action='store_true')
parser.add_argument('--use_5x5', default=False, action='store_true')
args = parser.parse_args()

n_warumup = 25
n_benchmark = 100

nb, nch, nz, nx, ny = args.bs, 1, 32, 216, 216

if args.blocks is None:

    blocks = [1,2,6,3]

else:
    blocks = [int(b) for b in args.blocks]

if args.more_filters:
    filters = [32, 64, 160, 384]
else:
    filters = 32

net = UNetResEncoderV2(in_channels=1, out_channels=3, is_2d=False,
                       z_to_xy_ratio=6.25,
                       n_blocks_list=blocks, filters=32,
                       use_5x5_on_full_res=args.use_5x5).cuda()

print(net)
xb = torch.zeros((nb, nch, nz, nx, ny), device='cuda')

context = nullcontext() if args.fp32 else torch.cuda.amp.autocast()

with context:
    for _ in range(n_warumup):
        out = net(xb)
        l = out[0].abs().mean()
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
        l = out[0].abs().mean()
        l.backward()
        net.zero_grad()
    et = perf_counter()
    print('{:.3e}'.format((et-st)/n_benchmark))
