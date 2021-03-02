from ovseg.data.SegmentationData import SegmentationData
import os
import matplotlib.pyplot as plt
from ovseg.augmentation.SegmentationAugmentation import SegmentationAugmentation
import torch
preprocessed_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'default')
keys = ['image', 'label']
folders = ['images', 'labels']
trn_dl_params = {'patch_size': [512, 512], 'batch_size': 4, 'epoch_len': 1,
                 'store_coords_in_ram': False}

data = SegmentationData(val_fold=0, preprocessed_path=preprocessed_path, keys=keys,
                        folders=folders, trn_dl_params=trn_dl_params)

# %%
gv_params1 = {'p_noise': 0, 'p_blur': 0, 'p_bright': 0, 'p_contr': 0}
gv_params2 = {'p_noise': 1, 'p_blur': 1, 'p_bright': 1, 'p_contr': 1}

aug_params1 = {'GPU_params': {'grayvalue': gv_params1}}
aug_params2 = {'GPU_params': {'grayvalue': gv_params2}}

aug1 = SegmentationAugmentation(**aug_params1)
aug2 = SegmentationAugmentation(**aug_params2)

# %%
for batch in data.trn_dl:
    break
batch = torch.cat([batch['image'], batch['label']], 1).cuda()
batch_np = batch.cpu().numpy()
# %%
batch_aug1 = aug1.GPU_augmentation.augment_batch(batch).cpu().numpy()
batch_aug2 = aug2.GPU_augmentation.augment_batch(batch).cpu().numpy()

# %%
plt.close('all')
for j in range(batch_np.shape[0]):
    plt.figure()
    for i, b in enumerate([batch_np, batch_aug1, batch_aug2]):
        plt.subplot(2, 3, i+1)
        plt.imshow(b[j, 0].astype(float), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 3, i+4)
        plt.imshow(b[j, 0, 192:320, 192:320].astype(float), cmap='gray')
        plt.axis('off')
