import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle


# %% first ovarian DSCs
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','OV04','pod_om_4fCV')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

P = np.load(os.path.join(predp, 'P_cross_validation_v2.npy'))
n_ens = 7

measures = {cl:{'gt':[], 'UQ_old':[], 'UQ_new':[]} for cl in [1,2,9]}

for ds_name in ['ApolloTCGA', 'BARTS']:
    
    scans = [s for s in os.listdir(os.path.join(predp, 'calibrated_0.00', ds_name+'_fold_5'))
             if s.endswith('.nii.gz')]

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
            if gt_cl.max()>0:
                DSC = 200*np.sum(gt_cl*pred)/np.sum(gt_cl+pred)
                measures[cl]['gt'].append(DSC)
                
                for hm, ext in zip([hm_old, hm_new], ['_old', '_new']):
                    
                    # UQ measures
                    sm = np.sum(hm+pred)
                    if sm > 0:
                        DSC_hat = 200 * np.sum(hm*pred)/sm
                    else:
                        DSC_hat = 100
                    
                    measures[cl]['UQ'+ext].append([DSC_hat])
                
# %% now kidney DSCS
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','kids21_trn','disease_3_1')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

P = np.load(os.path.join(predp, 'P_cross_validation_v2.npy'))
n_ens = 7

measures = {cl:{'gt':[], 'UQ_old':[], 'UQ_new':[]} for cl in [1,9]}

ds_name = 'kits21_tst'

scans = [s for s in os.listdir(os.path.join(rawp, ds_name, 'labels'))
         if s.endswith('.nii.gz')]

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
        if gt_cl.max()>0:
            DSC = 200*np.sum(gt_cl*pred)/np.sum(gt_cl+pred)
            measures[cl]['gt'].append(DSC)
            
            for hm, ext in zip([hm_old, hm_new], ['_old', '_new']):
                
                # UQ measures
                sm = np.sum(hm+pred)
                if sm > 0:
                    DSC_hat = 200 * np.sum(hm*pred)/sm
                else:
                    DSC_hat = 100
                
                measures[cl]['UQ'+ext].append([DSC_hat])
            
                
p = os.path.join(predp, 'UQ_measures.pkl')

pickle.dump(measures, open(p, 'wb'))