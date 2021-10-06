import nibabel as nib
from os import listdir, environ
from os.path import join
import numpy as np
from tqdm import tqdm
from time import sleep
gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')

w_list = [0.001, 0.01, 0.1]

for w in w_list:
    
    dscs = []
    sens_15 = []
    p1 = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'SLDS', 'U-Net5_{}'.format(w),
              'BARTS_ensemble_0_1_2_3_4')
    p2 = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'SLDS_reg_expert_{}'.format(w),
              'U-Net2','BARTS_ensemble_0_1_2_3_4')
    
    sleep(0.5)
    for case in tqdm(listdir(p1)):
        gt = nib.load(join(gtp, case)).get_fdata()
        gt_bin = (gt > 0).astype(float)
        gt_15 = (gt == 15).astype(float)
        pred1 = (nib.load(join(gtp, case)).get_fdata() == 1).astype(float)
        pred2 = (nib.load(join(gtp, case)).get_fdata() > 0 ).astype(float)
        pred = pred1 + pred2
        
        if pred.max() > 1:
            print('We got overlap for case {}'.format(case))
        
        if gt_bin.max() > 0:
            dscs.append(200 * np.sum(gt_bin * pred) / np.sum(gt_bin + pred))
        
        if gt_15.max() > 0:
            sens_15.append(100 * np.sum(gt_15 * pred) / np.sum(gt_15 + pred))
    print('w: {}, mean dsc: {:.2f}, sens 15: {:.2f}'.format(w, np.mean(dscs), np.mean(sens_15)))