from collections import defaultdict
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','OV04','pod_om_4fCV')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')


def zero():
    return 0

# %%
n_ens = 7
w_list = list(range(-3,4))

k_vec = np.zeros((2,2**7))
n_vec = np.zeros((2,2**7))

    
scans = [s for s in os.listdir(os.path.join(rawp, 'OV04','labels'))
         if s.endswith('.nii.gz')]

for scan in tqdm(scans):
    
    gt_seg = nib.load(os.path.join(rawp, 'OV04', 'labels', scan)).get_fdata()
    for c, cl in enumerate([1, 9]):
        gt = (gt_seg == cl).astype(float)
        preds = np.zeros_like(gt)
        
        for i, w in enumerate(w_list):
            
            model_name = f'calibrated_{w:.2f}'
            
            pred = nib.load(os.path.join(predp, model_name, 'cross_validation', scan)).get_fdata()
            pred = (pred == cl).astype(float)
            preds += 2**i * pred
        
        for i in range(2**7):
            
            I = preds == (i+1)
            
            n_vec[c,i] += np.sum(I.astype(int))
            k_vec[c,i] += np.sum(gt[I])


np.save(os.path.join(predp, 'k_cross_validation_v3.npy'), k_vec)
np.save(os.path.join(predp, 'n_cross_validation_v3.npy'), n_vec)
print(k_vec/n_vec)
