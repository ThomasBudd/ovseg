import numpy as np
import nibabel as nib
import os
from ovseg.utils.interp_utils import change_sample_pixel_spacing
from tqdm import tqdm

order = 1
spc_new = [0.6767578, 0.6767578, 5.]

rlp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')
plp = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_default', 'labels')


for case in tqdm(os.listdir(rlp)):
    img = nib.load(os.path.join(rlp, case))
    spc_old = img.header['pixdim'][1:4]
    im = img.get_fdata()
    sample = np.stack([im == 0, im == 9]).astype(float)
    sample = change_sample_pixel_spacing(sample, spc_old, spc_new, order)
    lb = np.argmax(sample, 0)
    np.save(os.path.join(plp, case[:8]+'.npy'), lb.astype(np.int8))
