from ovseg.utils.io import read_dcms
from ovseg.training.interpolation_by_overfitting import interpolation_by_overfitting
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import os

data_tpl = read_dcms('D:\\PhD\\Data\\ov_data_base\\raw_data\\OV04_dcm\\038\\CT_20091012')

im = data_tpl['image'].astype(float)
lb = (data_tpl['label'] == 9).astype(float)
im_rsz = resize(im, [213, 512, 512], order=3)
z = np.argmax(np.sum(lb, (1, 2)))

plt.subplot(1, 2, 1)
plt.imshow(im[z].clip(-150, 250),cmap='gray')
plt.contour(lb[z])
plt.subplot(1, 2, 2)
plt.imshow(im_rsz[z*2].clip(-150, 250),cmap='gray')
plt.contour(lb[z])

x = (im_rsz.clip(-150, 250) + 150) / 400
contains = np.where(np.sum(lb, (1, 2)) > 0)[0]
z_min, z_max = contains.min(), contains.max()

lb_crop = lb[z_min : z_max + 1]
im_crop = x[2*z_min : 2*z_max + 1]
# %%
lb_res = interpolation_by_overfitting(im_crop, lb_crop, target_DSC=99.5, max_iter=100,
                                      filters=8, base_lr=10**-4)