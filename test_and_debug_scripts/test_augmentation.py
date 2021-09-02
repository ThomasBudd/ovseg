from ovseg.augmentation.SegmentationAugmentation import SegmentationAugmentation
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.close('all')
torch_params = {}
torch_params['grid_inplane'] = {'p_rot': 1.0, 'p_zoom': 1.0, 'p_transl': 1.0, 'p_shear': 1.0,
                                'apply_flipping': False}
torch_params['grayvalue'] = {'p_noise': 1.0}

np_params = {'mask': {'p_morph': 1.0}}

aug = SegmentationAugmentation(torch_params=torch_params, np_params=np_params)


im_full = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04_test\\default\\images'
                  '\\OV04_034_20091014.npy')
lb_full = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04_test\\default\\labels'
                  '\\OV04_034_20091014.npy') > 0

im_crop = im_full[30:78, 100:292, 100:292].astype(np.float32)
imt = torch.from_numpy(im_crop).cuda().unsqueeze(0).unsqueeze(0).type(torch.float)
lb_crop = lb_full[30:78, 100:292, 100:292].astype(np.float32)
lbt = torch.from_numpy(lb_crop).cuda().unsqueeze(0).unsqueeze(0).type(torch.float)
xb = torch.cat([imt, lbt], 1).cuda()
xb = torch.cat([xb, xb], 0)

# %% test torch augmentation
xb_aug = aug.torch_augmentation(xb).cpu().numpy()

z = np.argmax(np.sum(lb_crop > 0, (1, 2)))
plt.subplot(2, 3, 1)
plt.imshow(xb_aug[0, 0, z], cmap='gray')
plt.contour(xb_aug[0, 1, z])
plt.subplot(2, 3, 2)
plt.imshow(im_crop[z], cmap='gray')
plt.contour(lb_crop[z])
plt.subplot(2, 3, 3)
plt.imshow(xb_aug[1, 0, z], cmap='gray')
plt.contour(xb_aug[1, 1, z])

# now numpy augmentation
xb_aug = aug.np_augmentation(xb.cpu().numpy())
z = np.argmax(np.sum(lb_crop > 0, (1, 2)))
plt.subplot(2, 3, 4)
plt.imshow(xb_aug[0, 0, z], cmap='gray')
plt.contour(xb_aug[0, 1, z])
plt.subplot(2, 3, 5)
plt.imshow(im_crop[z], cmap='gray')
plt.contour(lb_crop[z])
plt.subplot(2, 3, 6)
plt.imshow(xb_aug[1, 0, z], cmap='gray')
plt.contour(xb_aug[1, 1, z])
