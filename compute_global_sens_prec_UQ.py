import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle


# %% now the fun
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','OV04','pod_om_4fCV')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

n_ens = 7

ovlps_sens = np.zeros((2,3))
ovlps_prec = np.zeros((2,3))
vols_sens = np.zeros((2,3))
vols_prec = np.zeros((2,3))


ds_names = ['BARTS', 'ApolloTCGA']

for ds_name in ds_names:
    
    scans = [s for s in os.listdir(os.path.join(rawp, ds_name, 'labels'))
             if s.endswith('.nii.gz')]
    
    sleep(0.1)
    for scan in tqdm(scans):
        
        # get ground truht segmentation
        gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        
        # get dropout segmentations
        preds_drop = [nib.load(os.path.join(predp, 'dropout_UNet_0', f'{ds_name}_{f}_fold_5', scan)).get_fdata()
                     for f in range(7)]
        
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
        
            
            for pred in preds_cal:
                
                hm_new += 1/7 * (pred == cl).astype(float)
            
            # similarity measures
            
            for k, hm in enumerate([hm_new, hm_old, hm_drop]):
                
                pred_sens = (hm > 0).astype(float)
                
                ovlps_sens[c,k] += np.sum(pred_sens * gt_cl)
                vols_sens[c,k] += np.sum(gt_cl)
                
                pred_prec = (hm >= 0.99).astype(float)
                ovlps_prec[c,k] += np.sum(pred_prec * gt_cl)
                vols_prec[c,k] += np.sum(pred_prec)
            
            
sens = ovlps_sens / vols_sens
prec = ovlps_prec / vols_prec
p = os.path.join(predp, 'global_sens.pkl')
pickle.dump(sens, open(p, 'wb'))
p = os.path.join(predp, 'global_prec.pkl')
pickle.dump(prec, open(p, 'wb'))

# %%
sens = pickle.load(open(os.path.join(predp, 'global_sens.pkl'),'rb'))
print(sens)
prec = pickle.load(open(os.path.join(predp, 'global_prec.pkl'),'rb'))
print(prec)

# %%
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','kits21_trn','disease_3_1')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

n_ens = 7

ovlps_sens = np.zeros((1,3))
ovlps_prec = np.zeros((1,3))
vols_sens = np.zeros((1,3))
vols_prec = np.zeros((1,3))

ds_name = 'kits21_tst'

scans = [s for s in os.listdir(os.path.join(predp, 'UQ_calibrated_0.00', 'kits21_tst_ensemble_0_1_2'))
         if s.endswith('.nii.gz')]

sleep(0.1)
for scan in tqdm(scans):
    
    # get ground truht segmentation
    gt = nib.load(os.path.join(rawp, 'kits21', 'labels', scan)).get_fdata()
    
    # get dropout segmentations
    preds_drop = [nib.load(os.path.join(predp, 'dropout_UNet_0', f'kits21_tst_new_{f}_fold_3', scan)).get_fdata().astype(np.int16)
                 for f in range(7)]
    
    # get uncalibrated segmentations
    preds_unc = [nib.load(os.path.join(predp, 'UQ_calibrated_0.00', f'{ds_name}_fold_{f}', scan)).get_fdata().astype(np.int16)
                 for f in range(3,10)]
    
    # get calibrated segmentations
    preds_cal = [nib.load(os.path.join(predp,
                                       f'UQ_calibrated_{w:.2f}',
                                       f'{ds_name}_ensemble_0_1_2',
                                       scan)).get_fdata().astype(np.int16)
                 for w in range(-3,4)]        
    
    for c, cl in enumerate([2]):
    
        gt_cl = (gt == cl).astype(float)
        hm_drop = np.zeros_like(gt_cl, dtype=np.float16)
        hm_old = np.zeros_like(gt_cl, dtype=np.float16)
        hm_new = np.zeros_like(gt_cl, dtype=np.float16)
        
        # compute droput heatmap
        for pred in preds_drop:
            
            hm_drop += (pred == cl).astype(float)
        
        hm_drop = hm_drop/7
        
        # compute old heatmap
        for pred in preds_unc:
            
            hm_old += (pred == cl).astype(float)
        
        hm_old = hm_old/7
        
        # compute new heatmap        
        for pred in preds_cal:
            
            hm_new += 1/7 * (pred == cl).astype(float)
        
        
            
            for k, hm in enumerate([hm_new, hm_old, hm_drop]):
                
                pred_sens = (hm > 0).astype(float)
                
                ovlps_sens[c,k] += np.sum(pred_sens * gt_cl)
                vols_sens[c,k] += np.sum(gt_cl)
                
                pred_prec = (hm >= 0.99).astype(float)
                ovlps_prec[c,k] += np.sum(pred_prec * gt_cl)
                vols_prec[c,k] += np.sum(pred_prec)
        

sens = ovlps_sens / vols_sens
prec = ovlps_prec / vols_prec
p = os.path.join(predp, 'global_sens.pkl')
pickle.dump(sens, open(p, 'wb'))
p = os.path.join(predp, 'global_prec.pkl')
pickle.dump(prec, open(p, 'wb'))
# %%
sens = pickle.load(open(os.path.join(predp, 'global_sens.pkl'),'rb'))
print(sens)
prec = pickle.load(open(os.path.join(predp, 'global_prec.pkl'),'rb'))
print(prec)