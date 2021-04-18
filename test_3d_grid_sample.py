import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from scipy.ndimage import affine_transform

im_full = np.load('D:\PhD\Data\ov_data_base\preprocessed\OV04\pod_default\images_win_norm\\'
                  'case_000.npy')

im = np.moveaxis(im_full[:, :, 20:37], 2, 0)
# im = im_full[:, :, 20:37]
xb = np.stack([im[np.newaxis] for _ in range(2)])
imt = torch.from_numpy(xb).cuda()

theta = torch.zeros((2, 3, 4))
# scales = [0.9, 1.1]
# for i in range(3):
    # for j, s in enumerate(scales):
        # theta[j, i, i] = s
angles = [-5, 5]
i1, i2, i3 = 0, 1, 2
for i, angle in enumerate(angles):
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    theta[i, i1, i1] = c
    theta[i, i1, i2] = s
    theta[i, i2, i1] = -1*s
    theta[i, i2, i2] = c
    theta[i, i3, i3] = 1

grid = F.affine_grid(theta, imt.size()).cuda()

im_trsf = F.grid_sample(imt, grid).cpu().numpy()

z = 9
for i in range(3):
    plt.subplot(3, 3, 1 + 3*i)
    plt.imshow(im_trsf[0, 0, z - 1 + i], cmap='gray')
    plt.subplot(3, 3, 2 + 3*i)
    plt.imshow(im[z - 1 + i], cmap='gray')
    plt.subplot(3, 3, 3 + 3*i)
    plt.imshow(im_trsf[1, 0, z - 1 + i], cmap='gray')

# %% now again in scipy
theta_npy = np.zeros((2, 3, 3))
offsets = np.zeros((2, 3))
i1, i2, i3 = 2, 1, 0
# for i, angle in enumerate(angles):
#     c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
#     theta[i, i1, i1] = c
#     theta[i, i1, i2] = s
#     theta[i, i2, i1] = -1*s
#     theta[i, i2, i2] = c
#     theta[i, i3, i3] = 1
cen = np.array(im.shape) / 2
for i, s in enumerate([0.9, 1.1]):
    for j in range(3):
        theta_npy[i, j, j] = s
    offsets[i] = cen * (1 - s)
batch_list = []
for b in range(2):
    xb_trsf = np.stack([affine_transform(xb[b, c], theta_npy[b], offset=offsets[b])
                        for c in range(xb.shape[1])])
    batch_list.append(xb_trsf)
xb_trsf = np.stack(batch_list)
plt.figure()
for i in range(3):
    plt.subplot(3, 3, 1 + 3*i)
    plt.imshow(xb_trsf[0, 0, z - 1 + i], cmap='gray')
    plt.subplot(3, 3, 2 + 3*i)
    plt.imshow(xb[0, 0, z - 1 + i], cmap='gray')
    plt.subplot(3, 3, 3 + 3*i)
    plt.imshow(xb_trsf[1, 0, z - 1 + i], cmap='gray')
