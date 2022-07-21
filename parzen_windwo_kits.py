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

cl = 2

P = np.load(os.path.join(predp, 'P_cross_validation_v2.npy'))[:, 0]
P = np.concatenate([[0], P])

coefs = np.diff(P)

P = np.concatenate([P, [1]])
n_ens = 7
w_list = list(range(-3,4))

p_vs_p = {'P': P}

# %% parzen window function
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


# %% estimation on test set
ds_name = 'kits21_tst'
gtp = os.path.join(rawp, ds_name, 'labels')
scans = os.listdir(gtp)

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
        

p_vs_p['Test'] = N2 / N1
# %% estimation on CV
ds_name = 'kits21_trn'
gtp = os.path.join(rawp, ds_name, 'labels')
scans = os.listdir(gtp)

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
        pred = nib.load(os.path.join(predp, model_name, 'cross_validation', scan)).get_fdata()
        
        hm += coefs[i] * (pred == cl).astype(float)
    
    N1 += parzen_window(hm, scan in scans[:3])
    N2 += parzen_window(hm * seg)
        

p_vs_p['Cross-validation'] = N2 / N1

# %%
pickle.dump(p_vs_p, open(os.path.join(predp, 'p_vs_p_new.pkl'), 'wb'))
# %%

p_vs_p = pickle.load(open(os.path.join(predp, 'p_vs_p_new.pkl'), 'rb'))

# print(p_vs_p)

legend = ['Identity']

plt.close()
plt.plot([0, 1], [0, 1], 'k')
for key in p_vs_p:
    
    if key == 'P':
        continue
    
    legend.append(key)
    
    plt.plot(p_vs_p['P'][1:-1], p_vs_p[key][1:-1], 'o')
    
    SD = np.mean((p_vs_p['P'][1:-1] - p_vs_p[key][1:-1]) ** 2)
    print(f'{key} {SD:.3e}')


plt.legend(legend)