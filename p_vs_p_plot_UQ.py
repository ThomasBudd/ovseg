import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

predp = 'D:\\PhD\\Data\\ov_data_base\\predictions\\OV04\\pod_om_4fCV'
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

n_ens = 7
cl = 1

# %%

model_name = 'calibrated_0.00'
k_vec = np.zeros(n_ens)
n_vec = np.zeros(n_ens)

for ds_name in ['ApolloTCGA', 'BARTS']:
    
    scans = [s for s in os.listdir(os.path.join(predp, model_name, ds_name+'_fold_5'))
             if s.endswith('.nii.gz')]

    sleep(0.1)
    for scan in tqdm(scans):
        
        gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        gt = (gt == cl).astype(float)
        hm = np.zeros_like(gt)
        
        for f in range(5,5+n_ens):
            
            pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_fold_{f}', scan)).get_fdata()
            hm += (pred == cl).astype(float)
        
        for c in range(n_ens):
            I = (hm == c+1).astype(float)
            n_vec[c] += np.sum(I)
            k_vec[c] += np.sum(I*gt)

p_vec = k_vec/n_vec
np.save(os.path.join(predp, 'P_{n_ens}_bins_0_0_{cl}.npy'), p_vec)

# %%
all_models = [m for m in os.listdir(predp) if m.startswith('calibrated')]
all_w = np.array([float(m.split('_')[1]) for m in all_models])

dwmin, dwmax = -3, 3
w_list = np.linspace(dwmin, dwmax, n_ens)


k_vec = np.zeros(n_ens)
n_vec = np.zeros(n_ens)

for ds_name in ['ApolloTCGA', 'BARTS']:
    
    scans = [s for s in os.listdir(os.path.join(predp, model_name, ds_name+'_fold_5'))
             if s.endswith('.nii.gz')]

    sleep(0.1)
    for scan in tqdm(scans):
        
        gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        gt = (gt == cl).astype(float)
        hm = np.zeros_like(gt)
        
        for w in w_list:
            
            w_ind = np.argmin(np.abs(w-all_w))
            model_name = all_models[w_ind]
            
            pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_fold_5', scan)).get_fdata()
            hm += (pred == cl).astype(float)
        
        for c in range(n_ens):
            I = (hm == c+1).astype(float)
            n_vec[c] += np.sum(I)
            k_vec[c] += np.sum(I*gt)

p_vec = k_vec/n_vec
np.save(os.path.join(predp, f'P_{n_ens}_bins_{dwmin}_{dwmax}_{cl}.npy'), p_vec)

# %%
p_vec = np.load(os.path.join(predp, 'P_{n_ens}_bins_0_0_{cl}.npy'))
x = np.linspace(0, 1, n_ens+1)
plt.plot(x[1:], p_vec, 'bo')
p_vec = np.load(os.path.join(predp, f'P_{n_ens}_bins_{dwmin}_{dwmax}_{cl}.npy'))
x = np.linspace(0, 1, n_ens+1)
plt.plot(x[1:], p_vec, 'ro')
plt.plot([0, 1], [0, 1], 'k')
