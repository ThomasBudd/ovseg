import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','OV04','pod_om_4fCV')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

# %%
n_ens = 7
w_list = list(range(-3,4))

k_vec = np.zeros((n_ens, 2))
n_vec = np.zeros((n_ens, 2))

    
scans = [s for s in os.listdir(os.path.join(rawp, 'OV04','labels'))
         if s.endswith('.nii.gz')]

for scan in tqdm(scans):
    
    gt_seg = nib.load(os.path.join(rawp, 'OV04', 'labels', scan)).get_fdata()
    for i, cl in enumerate([1, 9]):
        gt = (gt_seg == cl).astype(float)
        preds = []
        
        for w in w_list:
            
            model_name = f'calibrated_{w:.2f}'
            
            pred = nib.load(os.path.join(predp, model_name, 'cross_validation', scan)).get_fdata()
            preds.append(pred)
        
        for c in range(n_ens-1):
            I = preds[c] - preds[c+1]
            n_vec[c, i] += np.sum(I)
            k_vec[c, i] += np.sum(I*gt)

        c = n_ens-1
        I = preds[c]
        
        n_vec[c, i] += np.sum(I)
        k_vec[c, i] += np.sum(I*gt)

p_vec = k_vec/n_vec
np.save(os.path.join(predp, 'P_cross_validation_v2.npy'), p_vec)
print(p_vec)