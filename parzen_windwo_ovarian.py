import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle

predp = os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'OV04',
                     'pod_om_4fCV')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

cl_list = ['1', '9']

P = np.load(os.path.join(predp, 'P_cross_validation_v2.npy'))
P = np.concatenate([np.zeros((1,2)), P], 0)

coefs = np.diff(P, axis=0)

P = np.concatenate([P, np.ones((1,2))], 0)
n_ens = 7
w_list = list(range(-3,4))

p_vs_p = {'P': P}

# %% parzen window function
def parzen_window(hm, i):
    
    vals, counts = np.unique(hm, return_counts=True)
    
    p = np.zeros_like(P)
    
    for val, count in zip(vals, counts):
        
        if val == 0:
            continue
        
        il = np.where(P < val)[0].max()
        xi = (val - P[il, i]) / (P[il+1, i] - P[il, i])
        
        p[il, i] += count * (1-xi)
        p[il+1, i] += count * xi

    return p
# %% estimation on test set

N1 = np.zeros_like(P)
N2 = np.zeros_like(P)
for ds_name in ['BARTS', 'ApolloTCGA']:
    gtp = os.path.join(rawp, ds_name, 'labels')
    scans = os.listdir(gtp)
    
    print(ds_name)
    sleep(0.1)
    for scan in tqdm(scans):
        
        gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        
        for i, cl in enumerate([1, 9]):
            seg = (gt == cl).astype(float)
            
            if seg.max() == 0:
                continue
            
            hm = np.zeros_like(seg)
                
            for j, w in enumerate(w_list):
                model_name = f'calibrated_{w:.2f}'
                pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_ensemble_0_1_2_3', scan)).get_fdata()
                
                hm += coefs[j] * (pred == cl).astype(float)
            
            N1 += parzen_window(hm, i)
            N2 += parzen_window(hm * seg, i)
        

p_vs_p['Test'] = N2 / N1
# %% estimation on CV
ds_name = 'OV04'
gtp = os.path.join(rawp, ds_name, 'labels')
scans = os.listdir(gtp)

N1 = np.zeros_like(P)
N2 = np.zeros_like(P)

for scan in tqdm(scans):
    
    gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()

    for i, cl in enumerate([1,9]):
        seg = (gt == cl).astype(float)
        
        if seg.max() == 0:
            continue
        
        hm = np.zeros_like(seg)
            
        for j, w in enumerate(w_list):
            model_name = f'UQ_calibrated_{w:.2f}'
            pred = nib.load(os.path.join(predp, model_name, 'cross_validation', scan)).get_fdata()
            
            hm += coefs[j] * (pred == cl).astype(float)
        
        N1 += parzen_window(hm, i)
        N2 += parzen_window(hm * seg, i)
            

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