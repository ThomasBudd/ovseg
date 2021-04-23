import torch
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
import numpy as np

im_full = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04\\pod_half\\images\\case_000.npy')
xb_3d = torch.from_numpy(im_full[np.newaxis, np.newaxis, 20:68, :192, :192]).type(torch.float)
xb_2d = torch.from_numpy(im_full[np.newaxis, np.newaxis, 21, :192, :192]).type(torch.float)

# %%
# fac = np.random.rand() + 1
fac = 2
xb_3d_lr = interpolate(xb_3d, scale_factor=[1/fac, 1/fac, 1/fac], mode='nearest')
xb_3d_lr = interpolate(xb_3d_lr, size=[48, 192, 192], mode='trilinear')
xb_2d_lr = interpolate(xb_2d, scale_factor=1/fac, mode='nearest')
xb_2d_lr = interpolate(xb_2d_lr, size=[192, 192], mode='bilinear')

# %%
plt.subplot(2, 2, 1)
plt.imshow(xb_3d[0, 0, 1].cpu().numpy(), cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(xb_3d_lr[0, 0, 1].cpu().numpy(), cmap='gray')
plt.subplot(2, 2, 3)
plt.imshow(xb_2d[0, 0].cpu().numpy(), cmap='gray')
plt.subplot(2, 2, 4)
plt.imshow(xb_2d_lr[0, 0].cpu().numpy(), cmap='gray')

