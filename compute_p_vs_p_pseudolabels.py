import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle


p_vs_p = {}
# %% parzen window function
def parzen_window(hm, P):
    
    vals, counts = np.unique(hm, return_counts=True)
    
    p = np.zeros_like(P)
    
    for val, count in zip(vals, counts):
        
        if val < P[0]:
            continue
        
        il = np.where(P < val)[0].max()
        xi = (val - P[il]) / (P[il+1] - P[il])
        
        p[il] += count * (1-xi)
        p[il+1] += count * xi

    return p

# %% first ovarian p vs p

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

p_vs_p['ovarian'] = {'P': P}

N1_old = np.zeros_like(P)
N2_old = np.zeros_like(P)

N1_new = np.zeros_like(P)
N2_new = np.zeros_like(P)

for ds_name in ['ApolloTCGA', 'BARTS']:
    
    gtp = os.path.join(rawp, ds_name, 'labels')
    scans = os.listdir(gtp)
    
    print(ds_name)
    sleep(0.1)
    for scan in tqdm(scans):
        
        # get ground truht segmentation
        gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        
        # get uncalibrated segmentations
        preds_unc = [nib.load(os.path.join(predp, 'calibrated_0.00', f'{ds_name}_fold_{f}', scan)).get_fdata()
                     for f in range(5,12)]
        
        # get calibrated segmentations
        preds_cal = [nib.load(os.path.join(predp,
                                           f'calibrated_{w:.2f}',
                                           f'{ds_name}_ensemble_0_1_2_3',
                                           scan)).get_fdata()
                     for w in range(-3,4)]        
        
        for c, cl in enumerate([1,9]):
        
            gt_cl = (gt == cl).astype(float)
            hm_old = np.zeros_like(gt_cl)
            hm_new = np.zeros_like(gt_cl)
            
            # compute old heatmap
            for pred in preds_unc:
                
                hm_old += (pred == cl).astype(float)
            
            hm_old = hm_old/7
            
            # compute new heatmap
            a_w_list = np.diff(np.concatenate([[0],P[:, c]]))
            
            for pred, a_w in zip(preds_cal, a_w_list):
                
                hm_new += a_w * (pred == cl).astype(float)
            
            # final prediction
            pred = (hm_old > 0.5).astype(float)
            
            # similarity measures
            N1_old[:,c] += parzen_window(hm_old, P[:, c])
            N2_old[:,c] += parzen_window(hm_old * gt_cl, P[:, c])
            
            N1_new[:,c] += parzen_window(hm_new, P[:, c])
            N2_new[:,c] += parzen_window(hm_new * gt_cl, P[:, c])

p_vs_p['ovarian']['Uncalibrated ensemble'] = N2_old/N1_old
p_vs_p['ovarian']['Calibrated ensemble'] = N2_old/N1_old

# %% now kidney DSCS
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','kids21_trn','disease_3_1')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

P = np.load(os.path.join(predp, 'P_cross_validation_v2.npy'))
P = np.concatenate([np.zeros((1,1)), P], 0)

coefs = np.diff(P, axis=0)

P = np.concatenate([P, np.ones((1,1))], 0)
n_ens = 7

ds_name = 'kits21_tst'

scans = [s for s in os.listdir(os.path.join(rawp, ds_name, 'labels'))
         if s.endswith('.nii.gz')]


p_vs_p['kidney'] = {'P': P}

N1_old = np.zeros_like(P)
N2_old = np.zeros_like(P)

N1_new = np.zeros_like(P)
N2_new = np.zeros_like(P)

sleep(0.1)
for scan in tqdm(scans):
    
    # get ground truht segmentation
    gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
    
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
        hm_old = np.zeros_like(gt_cl)
        hm_new = np.zeros_like(gt_cl)
        
        # compute old heatmap
        for pred in preds_unc:
            
            hm_old += (pred == cl).astype(float)
        
        hm_old = hm_old/7
        
        # compute new heatmap
        a_w_list = np.diff(np.concatenate([[0],P[:, c]]))
        
        for pred, a_w in zip(preds_cal, a_w_list):
            
            hm_new += a_w * (pred == cl).astype(float)
        
        # final prediction
        pred = (hm_old > 0.5).astype(float)
        
        # similarity measures
        
        N1_old[:,c] += parzen_window(hm_old, P[:, c])
        N2_old[:,c] += parzen_window(hm_old * gt_cl, P[:, c])
        
        N1_new[:,c] += parzen_window(hm_new, P[:, c])
        N2_new[:,c] += parzen_window(hm_new * gt_cl, P[:, c])
            
                
p_vs_p['kidney']['Uncalibrated ensemble'] = N2_old/N1_old
p_vs_p['kidney']['Calibrated ensemble'] = N2_old/N1_old


p = os.path.join(predp, 'p_vs_p_pseudolabels.pkl')

pickle.dump(p_vs_p, open(p, 'wb'))