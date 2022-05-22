import numpy as np
import os
import nibabel as nib
from tqdm import tqdm
from time import sleep

lbp = os.path.join(os.environ['OV_DATA_BASE'],'raw_data','OV04','labels')
predbp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 'OV04',
                      'pod_om_4fCV')

model_names = ['n_bias_1', 'new_loss_1', 'new_loss_0', 'new_loss_-1',
               'new_loss_-2', 'new_loss_-2_continued', 'new_loss_-2_continued_0'
               'new_loss_-2_continued_1']

all_metrics = {model_name:{'fp1':0, 'tp1':0, 'fn1':0, 'fp9':0, 'tp9':0, 'fn9':0} for model_name in model_names}

for case in tqdm(os.listdir(lbp)):
    gt = nib.load(os.path.join(lbp, case)).get_fdata()
   
    for model_name in model_names:
                
        predp = os.path.join(predbp, model_name, 'cross_validation')
        pred = nib.load(os.path.join(predp, case)).get_fdata()
        
        for cl in [1,9]:
            gt_c = (gt == cl).astype(float)
            pred_c = (pred == cl).astype(float)
            
            all_metrics[model_name][f'tp{cl}'] += np.sum(gt_c * pred_c)
            all_metrics[model_name][f'fp{cl}'] += np.sum((1-gt_c) * pred_c)
            all_metrics[model_name][f'fn{cl}'] += np.sum(gt_c * (1-pred_c))



# %%
for model_name in model_names:
    print(model_name)
    
    for cl in [1,9]:
        
        tp, fp, fn = all_metrics[model_name][f'tp{cl}'], all_metrics[model_name][f'fp{cl}'], all_metrics[model_name][f'fn{cl}']
        
        dsc = 100 * 2*tp / (2*tp + fp + fn)
        sens = 100 * tp / (tp + fn)
        prec = 100 * tp / (tp + fp)
        
        
        print(f'{cl}: {dsc:.2f}, {sens:.2f}, {prec:.2f}')
    