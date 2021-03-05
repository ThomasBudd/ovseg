from tqdm import tqdm
from ovseg.networks.recon_networks import get_operator
import numpy as np
import os
import torch

prep = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'kits19', 'default')
pp = os.path.join(prep, 'projections_normal')
fbpp = os.path.join(prep, 'fbp_normal')

op = get_operator()

for case in tqdm(os.listdir(pp)):
    p = np.load(os.path.join(pp, case))
    p = np.stack([p[np.newaxis, ..., z] for z in range(p.shape[-1])])
    proj = torch.from_numpy(p).cuda()
    filtered_sinogram = op.filter_sinogram(proj.to('cuda'))
    fbp = op.backprojection(filtered_sinogram).cpu().numpy()
    fbp = np.stack([fbp[z, 0] for z in range(fbp.shape[0])], -1)
    np.save(os.path.join(fbpp, case), fbp)
