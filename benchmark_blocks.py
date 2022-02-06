import torch
import torch.nn as nn
import argparse
from contextlib import nullcontext
from time import perf_counter


parser = argparse.ArgumentParser()
parser.add_argument("block", type=str)
parser.add_argument('--fp32', default=False, action='store_true')
args = parser.parse_args()

n_warumup = 25
n_benchmark = 100

nb, nch, nz, nx, ny = 1, 64, 32, 128, 128

if args.block == 'cnn':
    
    def Block():
        
        return nn.Sequential(*[nn.Conv3d(nch,nch,(1,3,3), padding=(0, 1, 1), bias=False),
                               nn.InstanceNorm3d(nch, affine=True),
                               nn.ReLU(),
                               nn.Conv3d(nch,nch,(1,3,3), padding=(0, 1, 1), bias=False),
                               nn.InstanceNorm3d(nch, affine=True),
                               nn.ReLU()])
elif args.block == 'dws':
    
    class Block(nn.Module):
        
        def __init__(self):
            super().__init__()
            
            self.conv = nn.Sequential(*[nn.Conv3d(nch,nch,(3,7,7), padding=(1, 3, 3), groups=nch, bias=False),
                                        nn.InstanceNorm3d(nch, affine=True),
                                        nn.ReLU(),
                                        nn.Conv3d(nch,nch,1, padding=0)])
        def forward(self, xb):
            
            return xb + self.conv(xb)

elif args.block == 'next':
    
    class Block(nn.Module):
        
        def __init__(self):
            super().__init__()
            
            self.conv = nn.Sequential(*[nn.Conv3d(nch,nch,(3,7,7), padding=(1, 3, 3), groups=nch, bias=False),
                                        nn.InstanceNorm3d(nch),
                                        nn.Conv3d(nch,nch*4,1, padding=0),
                                        nn.ReLU(),
                                        nn.Conv3d(nch*4,nch,1, padding=0)])
        def forward(self, xb):
            
            return xb + self.conv(xb)

else:
    raise NotImplementedError(f'Got unkown type of block {args.block}')


block = Block().cuda()
xb = torch.zeros((nb, nch, nz, nx, ny), device='cuda')

context = nullcontext() if args.fp32 else torch.cuda.amp.autocast()

with context:
    for _ in range(n_warumup):
        out = block(xb)
        l = out.abs().mean()
        l.backward()
    
    block.zero_grad()
    
    st = perf_counter()
    with torch.no_grad():
        for _ in range(n_benchmark):
            out = block(xb)
    et = perf_counter()
    
    print('{:.3e}'.format((et-st)/n_benchmark))
    st = perf_counter()
    for _ in range(n_benchmark):
        out = block(xb)
        l = out.abs().mean()
        l.backward()
        block.zero_grad()
    et = perf_counter()
    print('{:.3e}'.format((et-st)/n_benchmark))
