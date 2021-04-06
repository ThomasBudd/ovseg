import nibabel as nib
import numpy as np
import os
from ovseg.utils.io import read_nii

imp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'test_read_nii', 'images')

if not os.path.exists(imp):
    os.makedirs(imp)


def save_nii(im, sp, case):
    im_nii = nib.Nifti1Image(im, np.eye(4))
    im_nii.header['pixdim'][1:4] = sp
    nib.save(im_nii, os.path.join(imp, 'case_%03d.nii.gz' % int(case)))


shapes = [[128, 128, 64], [64, 128, 128], [128, 128, 96], [96, 128, 128], [128, 128, 128],
          [128, 128, 128]]
spacings = [[1, 1, 2], [2, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 2, 3]]

for case, (shape, sp) in enumerate(zip(shapes, spacings)):
    print('{}: save shape = {}, save spacing = {}'.format(case, shape, sp))
    im = np.random.rand(*shape)
    sp = np.array(sp)
    save_nii(im, sp, case)
    im_load, sp_load, _ = read_nii(os.path.join(imp, 'case_%03d.nii.gz' % int(case)))
    print('{}: load shape = {}, load spacing = {}'.format(case, im_load.shape, sp_load))

