import numpy as np
import nibabel as nib
import os
from time import sleep
from tqdm import tqdm
import pickle

dp = 'D:\\PhD\\Data\\nnUnet_raw_data_base\\RESULTS_FOLDER\\nnUNet\\3d_lowres\\Task120_OVPOD\\nnUNetTrainerV2__nnUNetPlansv2.1'
rawp = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\OV04\\labels'


dscs = []

for f in range(5):
    print(f, '\n')
    dscs_fold = []
    valp = os.path.join(dp, 'fold_'+str(f), 'validation_raw')
    sleep(0.5)
    cases = [case for case in os.listdir(valp) if case.endswith('.nii.gz')]
    for case in tqdm(cases):
        pred = nib.load(os.path.join(valp, case)).get_fdata()
        lb = nib.load(os.path.join(rawp, case)).get_fdata()
        pod = (lb == 9).astype(float)
        dscs_fold.append(200 * np.sum(pod * pred) / np.sum(pod + pred))
    dscs.append(np.array(dscs_fold))

all_dscs = []
for dsc in dscs[:-1]:
    all_dscs.extend(dsc)

# %%
val_cases = []
all_cases = []
for f in range(5):
    print(f, '\n')
    dscs_fold = []
    valp = os.path.join(dp, 'fold_'+str(f), 'validation_raw')
    sleep(0.5)
    cases = [case.split('.')[0] for case in os.listdir(valp) if case.endswith('.nii.gz')]
    val_cases.append(cases)
    all_cases.extend(cases)

splits = []
for f in range(5):
    val = val_cases[f]
    trn = [case for case in all_cases if case not in val]
    splits.append({'train': np.array(trn), 'val': np.array(val)})
pickle.dump(splits, open('splits_pod.pkl', 'wb'))