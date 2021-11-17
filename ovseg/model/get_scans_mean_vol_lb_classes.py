import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import pickle

lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04',
                   'labels')

lb_classes = [1,2,9,13,15,17]
has_fg = {i: [] for i in lb_classes}

for scan in tqdm(os.listdir(lbp)):
    
    img = nib.load(os.path.join(lbp, scan))
    seg = img.get_fdata()
    vvol = np.prod(img.header['pixdim'][1:4])/1000
    
    for cl in lb_classes:
        
        lb = (seg == cl).astype(float)
        if lb.max() > 0:
            has_fg[cl].append((np.sum(lb) * vvol, scan))

# %%
di = pickle.load(open(os.path.join(os.path.split(lbp)[0], 'data_info.pkl'), 'rb'))
for cl in lb_classes:
    
    vols = np.array([vol for vol, scan in has_fg[cl]])/1000
    scans = [scan for vol, scan in has_fg[cl]]
    meanvol = np.mean(vols)
    k = np.argmin(np.abs(vols - meanvol))
    vol = vols[k]
    scan = scans[k]
    pat_info = di[scan[:8]]
    
    print('cl: {}, meanvol: {:.2f}, scan: {}-{}, vol: {:.2f}'.format(cl,
                                                                     meanvol,
                                                                     pat_info['pat_id'],
                                                                     pat_info['timepoint'],
                                                                     vol))
    