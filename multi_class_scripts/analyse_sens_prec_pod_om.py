import numpy as np
import nibabel as nib
import os
from tqdm import tqdm

ds_name = 'BARTS'

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 
                     'OV04', 'pod_om_08_5', 'U-Net4_prg_lrn',
                     ds_name + '_ensemble_0_1_2_3_4')
# predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 
#                      'OV04', 'om_08', 'res_encoder_no_prg_lrn',
#                      ds_name + '_ensemble_0_1_2_3_4')
# predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 
#                      'OV04', 'pod_om_08_5', 'U-Net4_prg_lrn',
#                      'cross_validation')
gtp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
                   ds_name, 'labels')


def compute_metrics(gt, pred):
    
    if not gt.shape == pred.shape:
        print('Shape mismatch!')
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    gt_om = (gt == 1).astype(float)
    pred_om = (pred == 1).astype(float)

    if gt_om.max() > 0:
        ovlp = np.sum(gt_om * pred_om)
        vol_gt = np.sum(gt_om)
        vol_pred = np.sum(pred_om)
        
        dsc_om = 200 * ovlp / (vol_gt + vol_pred)
        sens_om = 100 * ovlp / vol_gt
        prec_om = 100 * ovlp / vol_pred if vol_pred > 0 else np.nan
    else:
        dsc_om, sens_om, prec_om = np.nan, np.nan, np.nan
    
    gt_pod = (gt == 9).astype(float)
    pred_pod = (pred == 9).astype(float)

    if gt_pod.max() > 0:
        ovlp = np.sum(gt_pod * pred_pod)
        vol_gt = np.sum(gt_pod)
        vol_pred = np.sum(pred_pod)
        
        dsc_pod = 200 * ovlp / (vol_gt + vol_pred)
        sens_pod = 100 * ovlp / vol_gt
        prec_pod = 100 * ovlp / vol_pred if vol_pred > 0 else np.nan
    else:
        dsc_pod, sens_pod, prec_pod = np.nan, np.nan, np.nan
    
    
    return dsc_om, sens_om, prec_om, dsc_pod, sens_pod, prec_pod


# %%

ovlp_om, ovlp_pod = 0, 0
vol_om, vol_pod = 0, 0
pvol_om, pvol_pod = 0, 0

metrics_list = []
for case in tqdm(os.listdir(predp)):
    
    gt = nib.load(os.path.join(gtp, case)).get_fdata()
    pred = nib.load(os.path.join(predp, case)).get_fdata()
    if not gt.shape == pred.shape:
        print('Shape mismatch!')
        continue
    metrics_list.append(compute_metrics(gt, pred))
    
    gt_om, gt_pod = (gt == 1).astype(float), (gt == 9).astype(float)
    pred_om, pred_pod = (pred == 1).astype(float), (pred == 9).astype(float)

    ovlp_om += np.sum(gt_om * pred_om)
    ovlp_pod += np.sum(gt_pod * pred_pod)

    vol_om += np.sum(gt_om)
    vol_pod += np.sum(gt_pod)
    
    pvol_om +=  np.sum(pred_om)
    pvol_pod +=  np.sum(pred_pod)

metrics_list = np.array(metrics_list)
mean_metrics = np.nanmean(metrics_list, 0)
print('mean metrics')
print(', '.join(['{:.1f}'.format(m) for m in mean_metrics]))

print('full metrics')
full_metrics = [200 * ovlp_om / (vol_om + pvol_om), 
                100 * ovlp_om / vol_om,
                100 * ovlp_om / pvol_om,
                200 * ovlp_pod / (vol_pod + pvol_pod),
                100 * ovlp_pod / vol_pod,
                100 * ovlp_pod / pvol_pod]
print(', '.join(['{:.1f}'.format(m) for m in full_metrics]))
