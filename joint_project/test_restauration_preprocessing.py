import nibabel as nib
import matplotlib.pyplot as plt
import os
from ovseg.preprocessing.Restauration2dSimPreprocessing import Restauration2dSimPreprocessing
import numpy as np
from ovseg.utils.io import read_dcms

prep = Restauration2dSimPreprocessing(n_angles=500,
                                      source_distance=600,
                                      det_count=736,
                                      det_spacing=1.0,
                                      # window=[-31.87, 318],
                                      # scaling=[52.2749, 38.163517],
                                      window=[-150, 250],
                                      scaling=[1, 0],
                                      target_z_spacing=5.0)

p = os.path.join(os.environ['OV_DATA_BASE'], 'fbp_for_subho')
data_tpl = read_dcms(os.path.join(p, 'DICOM'))
spacing = np.array(data_tpl['spacing'])
volume = data_tpl['image'].astype(float)
# change z axis first
# volume = np.moveaxis(volume, -1, 0)
# spacing = np.array([spacing[1], spacing[2], spacing[0]])
fbp, img = prep.preprocess_volume(volume, spacing)
real_fbp = np.moveaxis(np.load(os.path.join(p, 'fbp.npy')), -1, 0)
# %%
nz = img.shape[0]
plt.subplot(2, 3, 1)
plt.imshow(fbp[0],cmap='bone')
plt.subplot(2, 3, 4)
plt.imshow(img[0],cmap='bone')
plt.subplot(2, 3, 2)
plt.imshow(fbp[nz//2],cmap='bone')
plt.subplot(2, 3, 5)
plt.imshow(img[nz//2],cmap='bone')
plt.subplot(2, 3, 3)
plt.imshow(fbp[-1],cmap='bone')
plt.subplot(2, 3, 6)
plt.imshow(img[-1],cmap='bone')
mse = np.mean((img - fbp) ** 2)
Imax2 = (img.max() - img.min())**2
psnr = 10 * np.log10(Imax2 / mse)
print(psnr)

# %%
nz = img.shape[0]
plt.subplot(2, 3, 1)
plt.imshow(fbp[-1],cmap='bone')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(real_fbp[0],cmap='bone')
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(fbp[-4],cmap='bone')
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(real_fbp[3],cmap='bone')
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(fbp[-7],cmap='bone')
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(real_fbp[6],cmap='bone')
plt.axis('off')
