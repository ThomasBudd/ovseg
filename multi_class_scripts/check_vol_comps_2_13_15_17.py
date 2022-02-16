import os
from skimage.measure import label
import numpy as np
import nibabel as nib
from tqdm import tqdm

lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'labels')
lb_classes = [2, 13, 15, 17]

ccs = {str(cl): [] for cl in lb_classes}
vols = {str(cl): [] for cl in lb_classes}

for scan in tqdm(os.listdir(lbp)):
    
    img = nib.load(os.path.join(lbp, scan))
    fac = np.prod(img.header['pixdim'][1:4])
    lb = img.get_fdata()
    
    for cl in lb_classes:
        
        seg = (lb == cl).astype(float)
        
        if seg.max() > 0:
            vols[str(cl)].append(fac * np.sum(seg))
            ccs[str(cl)].append(label(seg).max())

# %%

