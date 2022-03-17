import numpy as np
import nibabel as nib
import os
from ovseg.data.Dataset import raw_Dataset
import tqdm


dcmp = 'D:\\PhD\\Data\\Apollo_Hilal'
dcm_ds = raw_Dataset(dcmp)
niip = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\ApolloTCGA'
nii_ds = raw_Dataset(niip)

dcs9 = []

for i in tqdm.tqdm(range(len(dcm_ds))):
    
    data_tpl_dcm = dcm_ds[i]
    data_tpl_nii = nii_ds[i]
    
    pod_dcm = (data_tpl_dcm['label'] == 9).astype(float)
    pod_nii = (data_tpl_nii['label'] == 9).astype(float)
    
    if pod_dcm.max() > 0:
        if pod_dcm.shape == pod_nii.shape:
            dcs9.append(200 * np.sum(pod_dcm * pod_nii) / np.sum(pod_dcm + pod_nii))
    