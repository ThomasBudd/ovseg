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

# %%

measures = {cl:{key:np.zeros(n_ens) for key in ['k_old', 'n_old',
                                                'k_new', 'n_new',
                                                'k_pred']} for cl in [1,9]}

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
        preds_cal = []
        
        w_list = list(range(-3,4))
        for w in w_list:
            model_name = f'calibrated_{w:.2f}'
            pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_fold_5', scan)).get_fdata()
            preds_cal.append(pred)
        
        for c, cl in enumerate([1,9]):
        
            gt_cl = (gt == cl).astype(float)
            hm_old = np.zeros_like(gt_cl)
            hm_new = np.zeros_like(gt_cl)
            
            # compute old heatmap
            for pred in preds_unc:
                
                hm_old += (pred == cl).astype(float)
            
            # compute new heatmap
            for pred in preds_cal:
                
                hm_new += (pred == cl).astype(float)
            
            # compute ensemble prediction
            pred_ens = (hm_old/n_ens > 0.5).astype(float)
            
            for e in range(n_ens):
                
                for hm, ext in zip([hm_old, hm_new], ['_old', '_new']):
                    lvl = (hm == e+1).astype(float)
                    measures[cl]['k'+ext][e] += np.sum(lvl*gt_cl)
                    measures[cl]['n'+ext][e] += np.sum(lvl)
                
                lvl = (hm_new == e+1).astype(float)
                measures[cl]['k_pred'][e] += np.sum(lvl*pred_ens)
            
                
p = os.path.join(predp, 'p_vs_p_measures.pkl')

pickle.dump(measures, open(p, 'wb'))