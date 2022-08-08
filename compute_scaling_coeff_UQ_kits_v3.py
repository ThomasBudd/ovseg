import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle
import torch

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','kits21_trn','disease_3_1')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

# %%
n_ens = 7
w_list = list(range(-3,4))

k_vec = np.zeros((2,128))
n_vec = np.zeros((2,128))

    
scans = [s for s in os.listdir(os.path.join(rawp, 'kits21_trn','labels'))
         if s.endswith('.nii.gz')]

for scan in tqdm(scans):
    
    gt_seg = nib.load(os.path.join(rawp, 'kits21_trn', 'labels', scan)).get_fdata()
    for c, cl in enumerate([2]):
        gt = (gt_seg == cl).astype(float)
        preds = np.zeros_like(gt)
        
        for i, w in enumerate(w_list):
            
            model_name = f'UQ_calibrated_{w:.2f}'
            
            pred = nib.load(os.path.join(predp, model_name, 'cross_validation', scan)).get_fdata()
            preds += 2**i * (pred == cl).astype(float)        

        gt = torch.from_numpy(gt).cuda()
        preds = torch.from_numpy(preds).cuda()
        
        for i in range(2**7):
            
            I = (preds == (i+1)).type(torch.int)
            
            n_vec[c,i] += torch.sum(I).item()
            k_vec[c,i] += torch.sum(gt * I).item()



np.save(os.path.join(predp, 'k_cross_validation_v3.npy'), k_vec)
np.save(os.path.join(predp, 'n_cross_validation_v3.npy'), n_vec)
print(k_vec/n_vec)

# %%
k_vec = np.load(os.path.join(predp, 'k_cross_validation_v3.npy'))
n_vec = np.load(os.path.join(predp, 'n_cross_validation_v3.npy'))

A_list = []
b_list = []

vec_list = []

for i7 in [0, 1]:
    for i6 in [0, 1]:
        for i5 in [0, 1]:
            for i4 in [0, 1]:
                for i3 in [0, 1]:
                    for i2 in [0, 1]:
                        for i1 in [0, 1]:
                            vec_list.append((i1, i2, i3, i4, i5, i6, i7))


for i, (k, n) in enumerate(zip(k_vec[0], n_vec[0])):
    
    if n > 0:
        A_list.append(np.array(vec_list[i+1]) * n)
        b_list.append(k)

A = np.array(A_list)
b = np.array(b_list)

A /= np.sum(b)
b /= np.sum(b)

coefs3 = np.linalg.lstsq(A,b)[0]

np.save(os.path.join(predp, 'coefs_v3.npy'),coefs3)

P1 = np.load(os.path.join(predp, 'P_cross_validation.npy'))[:, 0]
P2 = np.load(os.path.join(predp, 'P_cross_validation_v2.npy'))[:, 0]

coefs1 = np.diff(np.concatenate([[0], P1], 0))
coefs2 = np.diff(np.concatenate([[0], P2], 0))

print(coefs1)
print(coefs2)
print(coefs3)

# %%
import matplotlib.pyplot as plt
plt.close()
plt.plot(coefs1)
plt.plot(coefs2)
plt.plot(coefs3)

plt.legend(['1', '2', '3'])