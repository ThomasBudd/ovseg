import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle

# %% histogram function for old heatmaps

def my_hist(hm):

    return np.array([np.sum(hm == i/7) for i in range(1, 8)] + [1])

# %% parzen window
def parzen_window(hm):
    
    vals, counts = np.unique(hm, return_counts=True)
    
    P = np.arange(1, 9)/7
    
    p = np.zeros(P.shape[0])
    
    for val, count in zip(vals, counts):
        
        if val < P[0]:
            continue
        
        il = np.where(P < val)[0].max()
        xi = (val - P[il]) / (P[il+1] - P[il])
        
        p[il] += count * (1-xi)
        p[il+1] += count * xi
    
    return p

# %% now the fun
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','kits21_trn','disease_3_1')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

coefs = np.load(os.path.join(predp, 'coefs_v3.npy'))

n_ens = 7

N1_dict = {'gt_drop':np.zeros((n_ens+1, 2)), 'gt_old':np.zeros((n_ens+1, 2)),
               'gt_new':np.zeros((n_ens+1, 2)), 'pred_new':np.zeros((n_ens+1, 2))}
N2_dict = {'gt_drop':np.zeros((n_ens+1, 2)), 'gt_old':np.zeros((n_ens+1, 2)),
               'gt_new':np.zeros((n_ens+1, 2)), 'pred_new':np.zeros((n_ens+1, 2))}

measures = {cl:{'DSCs_gt':[], 'DSCs_drop':[], 'DSCs_old':[], 'DSCs_new':[]} for cl in [2]}

ds_name = 'kits21_tst'

scans = [s for s in os.listdir(os.path.join(predp, 'UQ_calibrated_0.00', 'kits21_tst_ensemble_0_1_2'))
         if s.endswith('.nii.gz')]

sleep(0.1)
for scan in tqdm(scans):
    
    # get ground truht segmentation
    gt = nib.load(os.path.join(rawp, 'kits21', 'labels', scan)).get_fdata()
    
    # get dropout segmentations
    preds_drop = [nib.load(os.path.join(predp, 'dropout_UNet_0', f'kits21_tst_new_{f}_fold_3', scan)).get_fdata()
                 for f in range(7)]
    
    # get uncalibrated segmentations
    preds_unc = [nib.load(os.path.join(predp, 'UQ_calibrated_0.00', f'{ds_name}_fold_{f}', scan)).get_fdata()
                 for f in range(3,10)]
    
    # get calibrated segmentations
    preds_cal = [nib.load(os.path.join(predp,
                                       f'UQ_calibrated_{w:.2f}',
                                       f'{ds_name}_ensemble_0_1_2',
                                       scan)).get_fdata()
                 for w in range(-3,4)]        
    
    for c, cl in enumerate([2]):
    
        gt_cl = (gt == cl).astype(float)
        hm_drop = np.zeros_like(gt_cl)
        hm_old = np.zeros_like(gt_cl)
        hm_new = np.zeros_like(gt_cl)
        
        # compute droput heatmap
        for pred in preds_drop:
            
            hm_drop += (pred == cl).astype(float)
        
        hm_drop = hm_drop/7
        
        # compute old heatmap
        for pred in preds_unc:
            
            hm_old += (pred == cl).astype(float)
        
        hm_old = hm_old/7
        
        # compute new heatmap
        a_w_list = coefs
        
        for pred, a_w in zip(preds_cal, a_w_list):
            
            hm_new += a_w * (pred == cl).astype(float)
        
        # final prediction
        pred = (hm_old > 0.5).astype(float)
        
        # similarity measures
        if gt_cl.max()>0:
            DSC = 200*np.sum(gt_cl*pred)/np.sum(gt_cl+pred)
            measures[cl]['DSCs_gt'].append(DSC)
            
            for hm, ext in zip([hm_drop, hm_old, hm_new], ['_drop', '_old', '_new']):
                
                # UQ measures
                sm = np.sum(hm+pred)
                if sm > 0:
                    DSC_hat = 200 * np.sum(hm*pred)/sm
                else:
                    DSC_hat = 100
                
                measures[cl]['DSCs'+ext].append([DSC_hat])
            
        # now p measures
        for hm, ext in zip([hm_drop, hm_old], ['_drop', '_old']):
                
            N1_dict['gt'+ext][:, c] += my_hist(hm)
            N2_dict['gt'+ext][:, c] += my_hist(hm*gt_cl)
            
            measures[cl]['P_gt'+ext] = N2_dict['gt'+ext][:, c]/N1_dict['gt'+ext][:, c]
            
                
        N1_dict['gt_new'][:, c] += parzen_window(hm_new)
        N2_dict['gt_new'][:, c] += parzen_window(hm_new*gt_cl)
        
        measures[cl]['P_gt_new'] = N2_dict['gt_new'][:, c]/N1_dict['gt_new'][:, c]
            
        # for pseudo label plot
        N1_dict['pred_new'][:, c] += parzen_window(hm_new)
        N2_dict['pred_new'][:, c] += parzen_window(hm_new*pred)
        
        # save in dict
        measures[cl]['P'] = np.arange(1, 9)/7
        measures[cl]['P_pred_new'] = N2_dict['pred_new'][:, c]/N1_dict['pred_new'][:, c]
        
        
p = os.path.join(predp, 'all_UQ_measures_v3.pkl')

pickle.dump(measures, open(p, 'wb'))