import numpy as np
import nibabel as nib
import os
from ovseg.utils.label_utils import reduce_classes

prev_stages = [{'data_name':'OV04',
                'preprocessed_name': 'pod_067',
                'model_name': 'larger_res_encoder'},
               {'data_name':'OV04',
                'preprocessed_name': 'om_067',
                'model_name': 'larger_res_encoder'},
               {'data_name': 'OV04',
                'preprocessed_name': 'multiclass_1_2_9',
                'model_name': 'U-Net5',
                'lb_classes': [2]},
               {'data_name': 'OV04',
                'preprocessed_name': 'multiclass_13_15_17',
                'model_name': 'U-Net5'}]

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions')
lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')

bin_dscs = []
bin_full_dscs = []

for case in os.listdir(lbp):
    
    gt = nib.load(os.path.join(lbp, case)).get_fdata()
    bin_gt = reduce_classes(gt, [1,2,9,13,15,17], True)
    full_bin_gt = (gt > 0).astype(float)

    prev_preds = []
    for ps in prev_stages:
        
        pred = nib.load(os.path.join(predp, ps['data_name'], ps['preprocessed_name'],
                                     ps['model_name'], 'BARTS_ensemble_0_1_2_3_4',
                                     case)).get_fdata()
        
        if 'lb_classes' in ps:
            pred_new = np.zeros_like(pred)
            for cl in ps['lb_classes']:
                pred_new[pred == cl] = 1
            prev_preds.append(pred_new)
        else:
            prev_preds.append((pred > 0).astype(float))

    bin_pred = np.max(np.stack(prev_preds), 0)
    
    if bin_gt.max() > 0:
        bin_dscs.append(200 * np.sum(bin_gt * bin_pred) / np.sum(bin_gt + bin_pred))
    if full_bin_gt.max() > 0:
        bin_full_dscs.append(200 * np.sum(full_bin_gt * bin_pred) / np.sum(full_bin_gt + bin_pred))

print('classes: {:.2f}, full: {:.2f}'.format(np.mean(bin_dscs), np.mean(bin_full_dscs)))
