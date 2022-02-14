import os
import numpy as np
import nibabel as nib
from skimage.measure import label
from tqdm import tqdm

lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'kits21',
                   'labels')

sizes = []

for case in tqdm(os.listdir(lbp)):
    
    
    img = nib.load(os.path.join(lbp, case))
    sp = img.header['pixdim'][1:4]
    lb = (img.get_fdata() == 2).astype(int)
    
    comps, n_comps = label(lb, return_num=True)
    
    for c in range(1, n_comps+1):
        comp = comps == c
        sizes.append([np.ptp(a)*s for a, s in zip(np.where(comp), sp)])

np.save(os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'kits21', 'tumour_sizes'), sizes)
