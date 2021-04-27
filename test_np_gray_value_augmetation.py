import numpy as np
from ovseg.augmentation.GrayValueAugmentation import np_gray_value_augmentation
import matplotlib.pyplot as plt

im = np.load('D:\PhD\Data\ov_data_base\preprocessed\OV04_test\default\images\\OV04_034_20091014.npy')

img = im[np.newaxis, np.newaxis, -30].astype(float)

aug = np_gray_value_augmentation()

param_dict = {'p_noise': [0, 1], 'p_bright': [0, 1], 'p_blur': [0, 0],
              'mm_var_noise': [np.array([0, 0.1]), np.array([0.1, 0.5])]}

# %%
plt.subplot(2, 2, 1)
plt.imshow(img[0, 0], cmap='gray')
aug.update_prg_trn(param_dict, 0.0)
plt.subplot(2, 2, 2)
plt.imshow(aug(img.copy())[0, 0], cmap='gray')
aug.update_prg_trn(param_dict, 0.5)
plt.subplot(2, 2, 3)
plt.imshow(aug(img.copy())[0, 0], cmap='gray')
aug.update_prg_trn(param_dict, 1.0)
plt.subplot(2, 2, 4)
plt.imshow(aug(img.copy())[0, 0], cmap='gray')

