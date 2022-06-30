import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','OV04','pod_om_4fCV')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

P = np.load(os.path.join(predp, 'P_cross_validation.npy'))

n_ens = 7
w_list = list(range(-3,4))

# %% use 100% models


# model_name = 'calibrated_0.00'
# gt_DSCs = [[], []]
# pred_DSCs = [[], []]

# for ds_name in ['ApolloTCGA', 'BARTS']:
    
#     scans = [s for s in os.listdir(os.path.join(predp, model_name, ds_name+'_fold_5'))
#              if s.endswith('.nii.gz')]

#     sleep(0.1)
#     for scan in tqdm(scans):
        
#         gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        
#         for i, cl in enumerate([1, 9]):

#             seg = (gt == cl).astype(float)
            
#             if seg.max() == 0:
#                 continue
            
#             hm = np.zeros_like(seg)
            
#             a_w_list = np.diff(np.concatenate([[0],P[:, i]]))
                
#             for w, a_w in zip(w_list,a_w_list):
#                 model_name = f'calibrated_{w:.2f}'
#                 pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_fold_5', scan)).get_fdata()
                
#                 hm += a_w * (pred == cl).astype(float)
            
#             pred = (hm > 0.5).astype(float)
            
#             # similarity measures
#             gt_DSCs[i].append(200*np.sum(seg*pred)/np.sum(seg+pred))
#             pred_DSCs[i].append(200 * np.sum(hm*pred)/np.sum(hm+pred))
            
# corr1_om = np.corrcoef(gt_DSCs[0], pred_DSCs[0])[0, 1]
# corr1_pod = np.corrcoef(gt_DSCs[1], pred_DSCs[1])[0, 1]
# DSC1_om = np.mean(gt_DSCs[0])
# DSC1_pod = np.mean(gt_DSCs[1])

# %% use 100% models new version


model_name = 'calibrated_0.00'
gt_DSCs = [[], []]
pred_DSCs = [[], []]

for ds_name in ['ApolloTCGA', 'BARTS']:
    
    scans = [s for s in os.listdir(os.path.join(predp, model_name, ds_name+'_fold_5'))
              if s.endswith('.nii.gz')]

    sleep(0.1)
    for scan in tqdm(scans):
        
        gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        
        for i, cl in enumerate([1, 9]):

            seg = (gt == cl).astype(float)
            
            if seg.max() == 0:
                continue
            
            hm = np.zeros_like(seg)
                
            for w in w_list:
                model_name = f'calibrated_{w:.2f}'
                pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_fold_5', scan)).get_fdata()
                
                hm += (pred == cl).astype(float)
            
            for j, p in enumerate(P[:, i]):
                hm[hm==j+1] = p
            
            pred = (hm > 0.5).astype(float)
            
            # similarity measures
            gt_DSCs[i].append(200*np.sum(seg*pred)/np.sum(seg+pred))
            pred_DSCs[i].append(200 * np.sum(hm*pred)/np.sum(hm+pred))
            
corr1_om = np.corrcoef(gt_DSCs[0], pred_DSCs[0])[0, 1]
corr1_pod = np.corrcoef(gt_DSCs[1], pred_DSCs[1])[0, 1]
DSC1_om = np.mean(gt_DSCs[0])
DSC1_pod = np.mean(gt_DSCs[1])
print('Omtentum')
print(corr1_om, DSC1_om)
print('POD')
print(corr1_pod, DSC1_pod)
# %% use ensemble predictions
# model_name = 'calibrated_0.00'
# gt_DSCs = [[], []]
# pred_DSCs = [[], []]

# for ds_name in ['ApolloTCGA', 'BARTS']:
    
#     scans = [s for s in os.listdir(os.path.join(predp, model_name, ds_name+'_fold_5'))
#              if s.endswith('.nii.gz')]

#     sleep(0.1)
#     for scan in tqdm(scans):
        
#         gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        
#         for i, cl in enumerate([1, 9]):

#             seg = (gt == cl).astype(float)
            
#             if seg.max() == 0:
#                 continue
            
#             hm = np.zeros_like(seg)
            
#             a_w_list = np.diff(np.concatenate([[0],P[:, i]]))
                
#             for w, a_w in zip(w_list,a_w_list):
#                 model_name = f'calibrated_{w:.2f}'
#                 pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_ensemble_0_1_2_3', scan)).get_fdata()
                
#                 hm += a_w * (pred == cl).astype(float)
            
#             pred = (hm > 0.5).astype(float)
            
#             # similarity measures
#             gt_DSCs[i].append(200*np.sum(seg*pred)/np.sum(seg+pred))
#             pred_DSCs[i].append(200 * np.sum(hm*pred)/np.sum(hm+pred))
            
# corr2_om = np.corrcoef(gt_DSCs[0], pred_DSCs[0])[0, 1]
# corr2_pod = np.corrcoef(gt_DSCs[1], pred_DSCs[1])[0, 1]
# DSC2_om = np.mean(gt_DSCs[0])
# DSC2_pod = np.mean(gt_DSCs[1])

# # %% use ensemble of ensembles
# gt_DSCs = [[], []]
# pred_DSCs = [[], []]

# for ds_name in ['ApolloTCGA', 'BARTS']:
    
#     scans = [s for s in os.listdir(os.path.join(predp, model_name, ds_name+'_fold_5'))
#              if s.endswith('.nii.gz')]

#     sleep(0.1)
#     for scan in tqdm(scans):
        
#         gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        
#         for i, cl in enumerate([1, 9]):

#             seg = (gt == cl).astype(float)
            
#             if seg.max() == 0:
#                 continue
            
#             hm = np.zeros_like(seg)
            
#             a_w_list = np.diff(np.concatenate([[0],P[:, i]]))
                
#             for w, a_w in zip(w_list,a_w_list):
#                 model_name = f'calibrated_{w:.2f}'
                
#                 for f in range(4):
#                     pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_fold_{f}', scan)).get_fdata()
                    
#                     hm += a_w/4 * (pred == cl).astype(float)
                
#             pred = (hm > 0.5).astype(float)
            
#             # similarity measures
#             gt_DSCs[i].append(200*np.sum(seg*pred)/np.sum(seg+pred))
#             pred_DSCs[i].append(200 * np.sum(hm*pred)/np.sum(hm+pred))
            
# corr3_om = np.corrcoef(gt_DSCs[0], pred_DSCs[0])[0, 1]
# corr3_pod = np.corrcoef(gt_DSCs[1], pred_DSCs[1])[0, 1]
# DSC3_om = np.mean(gt_DSCs[0])
# DSC3_pod = np.mean(gt_DSCs[1])

# %%
# print('Omentum')
# print(f'Corr: {corr1_om:.4f}, {corr2_om:.4f}, {corr3_om:.4f}')
# print(f'DSC: {DSC1_om:.4f}, {DSC2_om:.4f}, {DSC3_om:.4f}')
# print('POD')
# print(f'Corr: {corr1_pod:.4f}, {corr2_pod:.4f}, {corr3_pod:.4f}')
# print(f'DSC: {DSC1_pod:.4f}, {DSC2_pod:.4f}, {DSC3_pod:.4f}')


