import torch
from torch_radon import RadonFanbeam, Radon
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os

im = nib.load(os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'images',
                           'case_000_0000.nii.gz')).get_fdata()

im = np.moveaxis(im, -1, 0)[:, np.newaxis]
mu = 0.0192
# %% detmine n_angles
plt.close()
n_angles = 192
det_count = 724
source_distance = 595 * 1.4
det_distance = (1085.6-595) * 1.4
det_spacing = 1.4

# radon = RadonFanbeam(resolution=512,
#                       angles=np.linspace(0, 2 * np.pi, n_angles, endpoint=False),
#                       source_distance=source_distance,
#                       det_distance=det_distance,
#                       det_count=det_count,
#                       det_spacing=det_spacing)
radon = Radon(resolution=512,
              angles=np.linspace(0, np.pi, n_angles, endpoint=False),
              clip_to_circle=False,
              det_count=det_count)

def sim_sinogram(im_HU, num_photons=None):
    im_att = im_HU / 1000 * mu + mu
    proj = radon.forward(im_att)
    if num_photons is not None:
        proj_exp = torch.exp(-1 * proj)
        proj_exp = torch.poisson(proj_exp * num_photons) / num_photons
        proj = -1 * torch.log(proj_exp + 1e-6)
    return proj


def fbp(y):
    return radon.backprojection(radon.filter_sinogram(y))


def to_HU(im_att):
    return 1000 * (im_att - mu) / mu


y = sim_sinogram(torch.from_numpy(im.copy()).cuda().type(torch.float))
im_fbp = to_HU(fbp(y).cpu().numpy())

z_list = np.random.choice(list(range(im.shape[0])), size=3)
for i, z in enumerate(z_list):
    plt.subplot(2, 3, i+1)
    plt.imshow(im[z, 0].clip(-150, 250), cmap='gray')
    plt.subplot(2, 3, i+4)
    plt.imshow(im_fbp[z, 0].clip(-150, 250), cmap='gray')
plt.figure()
for i, z in enumerate(z_list):
    plt.subplot(2, 3, i+1)
    plt.imshow(im[z, 0, 192:320, 192:320].clip(-150, 250), cmap='gray')
    plt.subplot(2, 3, i+4)
    plt.imshow(im_fbp[z, 0, 192:320, 192:320].clip(-150, 250), cmap='gray')

# %%
plt.close()
n_photons = 2 * 10 ** 6
y = sim_sinogram(torch.from_numpy(im.copy()).cuda().type(torch.float), n_photons)
im_fbp = to_HU(fbp(y).cpu().numpy())

mse = np.mean((im_fbp.clip(-150, 250) - im.clip(-150, 250))**2)
Imax2 = 400**2
PSNR = 10 * np.log10(Imax2 / mse) if mse > 0 else np.nan
print(PSNR)
z_list = np.random.choice(list(range(im.shape[0])), size=3)
for i, z in enumerate(z_list):
    plt.subplot(2, 3, i+1)
    plt.imshow(im[z, 0].clip(-150, 250), cmap='gray')
    plt.subplot(2, 3, i+4)
    plt.imshow(im_fbp[z, 0].clip(-150, 250), cmap='gray')
plt.figure()
for i, z in enumerate(z_list):
    plt.subplot(2, 3, i+1)
    plt.imshow(im[z, 0, 192:320, 192:320].clip(-150, 250), cmap='gray')
    plt.subplot(2, 3, i+4)
    plt.imshow(im_fbp[z, 0, 192:320, 192:320].clip(-150, 250), cmap='gray')