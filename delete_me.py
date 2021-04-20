import numpy as np
import os
import nibabel as nib
from tqdm import tqdm
from time import sleep
bp = 'D:\\PhD\\Data\\nnUnet_raw_data_base\\RESULTS_FOLDER\\nnUNet\\3d_lowres\\Task120_OVPOD\\nnUNetTrainerV2__nnUNetPlansv2.1'

print(os.listdir(bp))

gtp = os.path.join(bp, 'gt_niftis')
pred_cases = []
for f in range(5):
    valp = os.path.join(bp, 'fold_%d' % f, 'validation_raw')
    pred_cases.extend([os.path.join(valp, case) for case in os.listdir(valp) if case.endswith('.gz')])

dscs = []
sleep(0.5)
for pred_case in tqdm(pred_cases):
    pred = nib.load(pred_case).get_fdata()
    gt = nib.load(os.path.join(gtp, os.path.basename(pred_case))).get_fdata()
    if gt.max() == 0:
        print(pred_case)
    dscs.append(200 * np.sum(gt * pred) / np.sum(gt + pred))

