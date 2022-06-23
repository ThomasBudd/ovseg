import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','OV04','pod_om_4fCV')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'OV04', 'pod_om_4fCV', 'heatmaps')

if not os.path.exists(plotp):
    os.makedirs(plotp)

P = np.load(os.path.join(predp, 'P_cross_validation.npy'))
n_ens = 7

for ds_name in ['ApolloTCGA', 'BARTS']:
    
    scans = [s for s in os.listdir(os.path.join(predp, 'calibrated_0.00', ds_name+'_fold_5'))
             if s.endswith('.nii.gz')]

    sleep(0.1)
    for scan in tqdm(scans[:10]):
        
        # get ground truht segmentation
        gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
        
        # get image
        im_name = [s for s in os.listdir(os.path.join(rawp, ds_name, 'iamges'))
                   if s.startswith(scan.split('.')[0])][0]
        im = nib.load(os.path.join(rawp, ds_name, 'images', im_name)).get_fdata()
        im = (im.clip(-150, 250) + 150)/400
        
        # get uncalibrated segmentations
        preds_unc = [nib.load(os.path.join(predp, 'calibrated_0.00', f'{ds_name}_fold_{f}', scan)).get_fdata()
                     for f in range(5,12)]
        
        # get calibrated segmentations
        preds_cal = [[] for _ in range(7)]
        
        for i, w in enumerate(list(range(-3,4))):
            model_name = f'calibrated_{w:.2f}'
            for f in range(4):
                
                pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_fold_{f}', scan)).get_fdata()
                preds_cal[i].append(pred)
        
        
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
            
            for i, a_w in enumerate(a_w_list):
                
                for pred in preds_cal[i]:
                    hm_new += a_w/4 * (pred == cl).astype(float)
            
            # final prediction
            pred = (hm_old > 0.5).astype(float)
            
            z_list = np.where(np.sum(gt_cl+hm_new+pred, (0,1)))[0]
            
            name = scan.split('.')[0]
            for z in z_list:
                
                plt.imshow(im[..., z]/2, cmap='gary', vmax=1)
                plt.contour(gt_cl[..., z] > 0, colors='blue')
                plt.axis('off')
                plt.savefig(os.path.join(plotp, f'{name}_{cl}_{z}_image.png'))
                plt.close()
                
                ovl = np.stack([im[..., z]/2+hm_new[...,z]/2, im[..., z]/2, im[..., z]/2], -1)
                plt.imshow(ovl)
                plt.contour(pred[..., z] > 0, colors='green')
                plt.axis('off')
                plt.savefig(os.path.join(plotp, f'{name}_{cl}_{z}_prediction.png'))
                plt.close()
                
                plt.imshow(hm_new[..., z], cmap='gary', vmax=1)
                plt.contour(gt_cl[..., z] > 0, colors='blue')
                plt.contour(pred[..., z] > 0, colors='green')
                plt.axis('off')
                plt.savefig(os.path.join(plotp, f'{name}_{cl}_{z}_heatmap.png'))
                plt.close()
                