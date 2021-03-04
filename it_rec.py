import torch
import numpy as np
from ovseg.networks.recon_networks import get_operator
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filter", required=False, default="ramp")
args = parser.parse_args()


def PSNR(im_gt, im_it):
    Im2 = (im_gt.min() - im_gt.max()) ** 2
    mse = torch.square(im_gt - im_it).mean()
    return 10 * np.log10(Im2.item() / mse.item())

def fit(Ax, y):
    return torch.square(Ax-y).mean().item()

def fbp(y):
    return op.backwards(op.filter_sinogram(y, args.filter))

n_iters = 10
op = get_operator()

proj = np.load(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_default',
                            'projections_HU', 'case_000.npy'))
im = np.load(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_default',
                          'images_HU_rescale', 'case_000.npy'))

y = torch.from_numpy(proj[..., 0]).cuda()
x_star = torch.from_numpy(im[..., 0]).cuda()

x = fbp(y)
Ax = op.forward(x)
print('It 0: PSNR: {:.3f}, fit: {:.4f}'.format(PSNR(x_star, x), fit(Ax, y)))

for i in range(1, n_iters + 1):
    delta_x = fbp(y - Ax)
    Adelta_x = op.forward(delta_x)
    lambd = (Adelta_x * (y - Ax)).mean() / (Adelta_x * Adelta_x).mean()
    x = x + lambd * delta_x
    Ax = op.forward(x)
    print('It {}: PSNR: {:.3f}, fit: {:.4f}'.format(i, PSNR(x_star, x), fit(Ax, y)))
