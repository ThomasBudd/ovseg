import torch
import numpy as np
from ovseg.networks.recon_networks import get_operator
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--filter", required=False, default="hann")
args = parser.parse_args()


def PSNR(im_gt, im_it):
    Im2 = (im_gt.min() - im_gt.max()) ** 2
    mse = torch.square(im_gt - im_it).mean()
    return 10 * np.log10(Im2.item() / mse.item())


def fit(Ax, y):
    return torch.square(Ax-y).mean().item()


def fbp(y):
    return op.backprojection(op.filter_sinogram(y, args.filter))


n_iters = 2
op = get_operator()

for i in range(276, 281):

    proj = np.load(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_default',
                                'projections_high_0', 'case_'+i+'.npy'))
    im = np.load(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_default',
                              'images_HU_rescale', 'case_'+i+'.npy'))
    proj, im = np.moveaxis(proj[np.newaxis], -1, 0), np.moveaxis(im[np.newaxis], -1, 0)
    y = torch.from_numpy(proj).cuda()
    x_star = torch.from_numpy(im).cuda()

    x_fbp = fbp(y)
    x = x_fbp
    Ax = op.forward(x)
    print('It 0: PSNR: {:.3f}, fit: {:.8f}'.format(PSNR(x_star, x), fit(Ax, y)))

    for i in range(1, n_iters + 1):
        delta_x = fbp(y - Ax)
        Adelta_x = op.forward(delta_x)
        lambd = (Adelta_x * (y - Ax)).mean() / (Adelta_x * Adelta_x).mean()
        x = x + lambd * delta_x
        Ax = op.forward(x)
        print('It {}: PSNR: {:.3f}, fit: {:.8f}'.format(i, PSNR(x_star, x), fit(Ax, y)))
