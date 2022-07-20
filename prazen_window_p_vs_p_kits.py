import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle

predp = os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'kits21_trn',
                     'disease_3_1')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

ds_name = 'kits21_tst'
gtp = os.path.join(rawp, ds_name, 'labels')
scans = os.listdir(gtp)
cl = 2

P = np.load(os.path.join(predp, 'P_cross_validation.npy'))[:, 0]
P = np.concatenate([[0], P])

coefs = np.diff(P)

P = np.concatenate([P, [1]])
n_ens = 7
w_list = list(range(-3,4))

p_vs_p = {'P': P}

# %% parzen window
def parzen_window(hm, prnt_sums=False):
    
    vals, counts = np.unique(hm, return_counts=True)
    
    p = np.zeros_like(P)
    
    for val, count in zip(vals, counts):
        
        if val == 0:
            continue
        
        il = np.where(P < val)[0].max()
        xi = (val - P[il]) / (P[il+1] - P[il])
        
        p[il] += count * (1-xi)
        p[il+1] += count * xi
    
    if prnt_sums:
        print(np.sum(counts[vals > 0]), np.sum(p))
    
    return p

# %% use 100% ensemble
gt_DSCs = [[], []]
pred_DSCs = [[], []]

N1 = np.zeros_like(P)
N2 = np.zeros_like(P)

for scan in tqdm(scans):
    
    gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        

    seg = (gt == cl).astype(float)
    
    if seg.max() == 0:
        continue
    
    hm = np.zeros_like(seg)
        
    for i, w in enumerate(w_list):
        model_name = f'UQ_calibrated_{w:.2f}'
        pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_ensemble_3_4_5', scan)).get_fdata()
        
        hm += coefs[i] * (pred == cl).astype(float)
    
    N1 += parzen_window(hm, scan in scans[:3])
    N2 += parzen_window(hm * seg)
    

p_vs_p['P_ensemble_3_4_5'] = N2 / N1

# %% use CV ensemble
model_name = 'UQ_calibrated_0.00'
gt_DSCs = [[], []]
pred_DSCs = [[], []]

N1 = np.zeros_like(P)
N2 = np.zeros_like(P)

for scan in tqdm(scans):
    
    gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        

    seg = (gt == cl).astype(float)
    
    if seg.max() == 0:
        continue
    
    hm = np.zeros_like(seg)
        
    for i, w in enumerate(w_list):
        model_name = f'UQ_calibrated_{w:.2f}'
        pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_ensemble_0_1_2', scan)).get_fdata()
        
        hm += coefs[i] * (pred == cl).astype(float)
    
    N1 += parzen_window(hm, scan in scans[:3])
    N2 += parzen_window(hm * seg)
        

p_vs_p['P_ensemble_0_1_2'] = N2 / N1


# %% use CV folds ensemble
model_name = 'UQ_calibrated_0.00'
gt_DSCs = [[], []]
pred_DSCs = [[], []]

N1 = np.zeros_like(P)
N2 = np.zeros_like(P)

for scan in tqdm(scans):
    
    gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        

    seg = (gt == cl).astype(float)
    
    if seg.max() == 0:
        continue
    
    hm = np.zeros_like(seg)
        
    for i, w in enumerate(w_list):
        model_name = f'UQ_calibrated_{w:.2f}'
        
        for f in range(3):
            pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_fold_{f}', scan)).get_fdata()
            hm += coefs[i]/3 * (pred == cl).astype(float)
    
    N1 += parzen_window(hm, scan in scans[:3])
    N2 += parzen_window(hm * seg)
        

p_vs_p['P_folds_0_1_2'] = N2 / N1

# %%
pickle.dump(p_vs_p, open(os.path.join(predp, 'p_vs_p.pkl'), 'wb'))
# %%

p_vs_p = pickle.load(open(os.path.join(predp, 'p_vs_p.pkl'), 'rb'))

# print(p_vs_p)


plt.close()
plt.plot([0, 1], [0, 1], 'k')
for key in ['P_ensemble_3_4_5', 'P_ensemble_0_1_2', 'P_folds_0_1_2']:
    
    plt.plot(p_vs_p['P'][1:-1], p_vs_p[key][1:-1], 'o')
    
    SD = np.mean((p_vs_p['P'][1:-1] - p_vs_p[key][1:-1]) ** 2)
    print(f'{key} {SD:.3e}')


plt.legend(['Identity', 'P_ensemble_3_4_5', 'P_ensemble_0_1_2', 'P_folds_0_1_2'])