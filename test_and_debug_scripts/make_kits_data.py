from os import listdir, makedirs, environ
from os.path import join, exists
from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.utils.io import save_nii
from tqdm import tqdm
import nibabel as nib
import numpy as np
from ovseg.io.interp_utils import change_img_pixel_spacing

copy_data = True
kits_path = '/local/scratch/public/tb588/kits19/data'
ov_raw = join(environ['OV_DATA_BASE'], 'raw_data', 'kits19')

for subf in ['images', 'labels']:
    if not exists(join(ov_raw, subf)):
        makedirs(join(ov_raw, subf))

if copy_data:
    for case in tqdm(listdir(kits_path)):
        if exists(join(kits_path, case, 'segmentation.nii.gz')):
            # first for the segmentation
            img = nib.load(join(kits_path, case, 'segmentation.nii.gz'))
            im = np.swapaxes(img.get_fdata(), 0, -1)
            spacing = img.header['pixdim'][[2, 3, 1]]
            spacing_new = [spacing[0], spacing[1], 3]
            im = change_img_pixel_spacing(im, spacing, spacing_new, 0)
            out_file = join(ov_raw, 'labels', 'case_{}.nii.gz'.format(case[-3:]))
            save_nii(im, out_file, spacing=spacing_new)
            # now again for the image
            img = nib.load(join(kits_path, case, 'imaging.nii.gz'))
            im = np.swapaxes(img.get_fdata(), 0, -1)
            im = change_img_pixel_spacing(im, spacing, spacing_new, 3)
            out_file = join(ov_raw, 'images', 'case_{}_0000.nii.gz'.format(case[-3:]))
            save_nii(im, out_file, spacing=spacing_new)


# %% now preprocessing
raw_data = 'kits19'
preprocessing = SegmentationPreprocessing(window=[-79, 304], scaling=[76.69, 101.0])
preprocessing.plan_preprocessing_raw_data(raw_data)
preprocessing.preprocess_raw_data(raw_data)