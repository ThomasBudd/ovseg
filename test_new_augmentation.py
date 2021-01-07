from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
import matplotlib.pyplot as plt
import numpy as np
import torch

model_parameters = get_model_params_2d_segmentation()
model_parameters['data']['trn_dl_params']['store_coords_in_ram'] = False
model_parameters['data']['val_dl_params']['store_coords_in_ram'] = False
model_parameters['augmentation']['CPU_params'] = {'mask': {'p_morph': 1}}
model_parameters['augmentation']['TTA_params'] = {'spatial': model_parameters['augmentation']['GPU_params']['spatial']}


model = SegmentationModel(val_fold=0, data_name='all', model_name='delete_me',
                          model_parameters=model_parameters)

# %%
data = model.data.trn_ds[0]
volume = np.stack([data['image'], data['label']])
imt = torch.from_numpy(volume).cuda()
im_aug1 = model.augmentation.GPU_augmentation.augment_volume(torch.from_numpy(volume.copy()).cuda()).cpu().numpy()
im_aug2 = model.augmentation.CPU_augmentation.augment_volume(volume.copy())
im_aug3 = model.augmentation.TTA.augment_volume(torch.from_numpy(volume.copy()).cuda()).cpu().numpy()

def plt_im(volume):
    im, lb = volume
    z = np.argmax(np.sum(lb, (0,1)))
    plt.imshow(im[..., z], cmap='gray')
    plt.contour(lb[..., z] > 0)

for i, vol in enumerate([volume, im_aug1, im_aug2, im_aug3]):
    plt.subplot(2, 2, i+1)
    plt_im(vol)