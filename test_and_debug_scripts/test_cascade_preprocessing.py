from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
import os
import nibabel as nib
import numpy as np
from ovseg.data.Dataset import Dataset
from ovseg.data.SegmentationDataloader import SegmentationDataloader
from ovseg.augmentation.MaskAugmentation import MaskAugmentation
import matplotlib.pyplot as plt

# %%
dp = 'D:\\PhD\\Data\\ov_data_base\\predictions\\Test\\test_test\\test_cascade\\cross_validation'

for case in os.listdir(dp):
    img = nib.load(os.path.join(dp, case))
    lb = img.get_fdata()
    print(np.unique(lb))
    om = (lb == 1).astype(float)
    if om.max() > 0:
        nii_img = nib.Nifti1Image(om, img.affine, img.header)
        nib.save(nii_img, os.path.join(dp, case))
    else:
        print(case)

# %%

model_params = get_model_params_2d_segmentation()

for f in range(5):
    model = SegmentationModel(val_fold=f, data_name='Test', preprocessed_name='test_test',
                              model_name='test_cascade', model_parameters=model_params)
    model.preprocess_prediction_for_next_stage('test_test')

# %%
scans = ['case_%03d.npy'%i for i in [0, 1, 2, 3, 4, 7, 8]]
preprocessed_path = 'D:\\PhD\\Data\\ov_data_base\\preprocessed\\Test\\test_test'
keys = ['image', 'label', 'pred_fps']
folders = ['images', 'labels', 'test_test_test_cascade']
ds = Dataset(scans, preprocessed_path, keys, folders)
augmentation = MaskAugmentation([5.       , 0.6777344, 0.6777344], p_morph=1.0)
dl = SegmentationDataloader(vol_ds=ds,patch_size=[48, 192, 192], batch_size=1, num_workers=None,
                            pin_memory=True, epoch_len=250, p_bias_sampling=0,
                            min_biased_samples=1, augmentation=augmentation, padded_patch_size=None,
                            store_coords_in_ram=True, memmap='r', n_im_channels=1,
                            image_key='image',
                            label_key='label', pred_fps_key='pred_fps', n_fg_classes=1,
                            store_data_in_ram=False, return_fp16=False, n_max_volumes=None)

# %%
for batch in dl:
    break
batch = batch.cpu().numpy()[0]
z = np.argmax(np.sum(batch[2], (1, 2)))
batch = batch[:, z]
plt.imshow(batch[0], cmap='gray')
plt.contour(batch[1], colors='blue', linewidth=0.5, linestyles='dashed')
plt.contour(batch[2], colors='red', linewidth=0.5, linestyles='dotted')