import nibabel as nib
import matplotlib.pyplot as plt
import os
from ovseg.preprocessing.Restauration2dSimPreprocessing import Restauration2dSimPreprocessing
import numpy as np

prep = Restauration2dSimPreprocessing(n_angles=500,
                                      source_distance=600,
                                      det_count=736,
                                      det_spacing=1.0,
                                      num_photons=10**6,
                                      window=[-31.87, 318],
                                      scaling=[52.2749, 38.163517],
                                      target_z_spacing=5.0)
nib_img = nib.load(os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'images',
                                'case_000_0000.nii.gz'))
spacing = nib_img.header['pixdim'][1:4]
volume = nib_img.get_fdata()
# change z axis first
volume = np.moveaxis(volume, -1, 0)
spacing = np.array([spacing[1], spacing[2], spacing[0]])
fbp, img = prep.preprocess_volume(volume, spacing)
# %%
nz = img.shape[0]
plt.subplot(2, 3, 1)
plt.imshow(fbp[0],cmap='gray')
plt.subplot(2, 3, 4)
plt.imshow(img[0],cmap='gray')
plt.subplot(2, 3, 2)
plt.imshow(fbp[nz//2],cmap='gray')
plt.subplot(2, 3, 5)
plt.imshow(img[nz//2],cmap='gray')
plt.subplot(2, 3, 3)
plt.imshow(fbp[-1],cmap='gray')
plt.subplot(2, 3, 6)
plt.imshow(img[-1],cmap='gray')
mse = np.mean((img - fbp) ** 2)
Imax2 = (img.max() - img.min())**2
psnr = 10 * np.log10(Imax2 / mse)
print(psnr)


# %%
prep.preprocess_raw_folders('OV04', 'pod_2d')