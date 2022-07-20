import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','kits21_trn','disease_3_1')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

# %%
n_ens = 7
w_list = list(range(-3,4))

k_vec = np.zeros((n_ens, 2))
n_vec = np.zeros((n_ens, 2))

    
scans = [s for s in os.listdir(os.path.join(rawp, 'kits21_trn','labels'))
         if s.endswith('.nii.gz')]

for scan in tqdm(scans):
    
    gt_seg = nib.load(os.path.join(rawp, 'kits21_trn', 'labels', scan)).get_fdata()
    for i, cl in enumerate([2, 3]):
        gt = (gt_seg == cl).astype(float)
        hm = np.zeros_like(gt)
        
        for w in w_list:
            
            model_name = f'UQ_calibrated_{w:.2f}'
            
            pred = nib.load(os.path.join(predp, model_name, 'cross_validation', scan)).get_fdata()
            hm += (pred == cl).astype(float)
        
        for c in range(n_ens):
            I = (hm == c+1).astype(float)
            n_vec[c, i] += np.sum(I)
            k_vec[c, i] += np.sum(I*gt)

p_vec = k_vec/n_vec
np.save(os.path.join(predp, 'P_cross_validation.npy'), p_vec)
# %%
p_vec = np.load(os.path.join(predp, 'P_cross_validation.npy'))
print(p_vec)