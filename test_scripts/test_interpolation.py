import numpy as np
from ovseg.utils.interp_utils import change_img_pixel_spacing, torch_interp_img
from ovseg.utils import grid_utils
import matplotlib.pyplot as plt
import nibabel as nib
import os
import torch
plt.close('all')

OV04_path = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04')
imp = os.path.join(OV04_path, 'images')
img = nib.load(os.path.join(imp, os.listdir(imp)[0]))
spc_old = img.header['pixdim'][1:4]
img = img.get_fdata()
img_gpu = torch.from_numpy(img).cuda()

# %% test resizing
spc_new = [0.7, 0.7, 5]
img_rsz = change_img_pixel_spacing(img_gpu, spc_old, spc_new, order=1).cpu().numpy()

# %%
grid = grid_utils.get_centred_torch_grid(img.shape, device='cuda', spacing=spc_old)
grid = grid_utils.rotate_grid_3d(grid, (0.1, 0.0, 0.5)) * 0.9
grid = grid_utils.grid_to_indices(grid, img.shape, spacing=spc_old)

img_3daug = torch_interp_img(img_gpu, grid, order=1).cpu().numpy()

# %%
grid = grid_utils.get_centred_torch_grid(img.shape[:2], device='cuda', spacing=spc_old[:2])
grid = grid_utils.rotate_grid_2d(grid, 0.5) * 1.1
grid = grid_utils.grid_to_indices(grid, img.shape[:2], spacing=spc_old[:2])

img_2daug = torch_interp_img(img_gpu, grid, order=1).cpu().numpy()

# %%
plt.figure()
for i, z  in enumerate([0, 25, 60]):
    plt.subplot(2, 3, i+1)
    plt.imshow(img[..., z], cmap='gray', vmin=-150, vmax=250)
    plt.subplot(2, 3, i+4)
    plt.imshow(img_rsz[..., z], cmap='gray', vmin=-150, vmax=250)

# %%
plt.figure()
for i, z  in enumerate([0, 25, 60]):
    plt.subplot(2, 3, i+1)
    plt.imshow(img[..., z], cmap='gray', vmin=-150, vmax=250)
    plt.subplot(2, 3, i+4)
    plt.imshow(img_3daug[..., z], cmap='gray', vmin=-150, vmax=250)
# %%
plt.figure()
for i, z  in enumerate([0, 25, 60]):
    plt.subplot(2, 3, i+1)
    plt.imshow(img[..., z], cmap='gray', vmin=-150, vmax=250)
    plt.subplot(2, 3, i+4)
    plt.imshow(img_2daug[..., z], cmap='gray', vmin=-150, vmax=250)


