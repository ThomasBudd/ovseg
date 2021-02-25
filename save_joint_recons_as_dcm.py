import numpy as np
from os import listdir, mkdir
from os.path import join, exists
from shutil import copyfile
import nibabel as nib
import pydicom
import pickle

nii_path = 'D:\\PhD\\Data\\ov_data_base\\learned_recons\\joined_win'
aptc_path = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\ApolloTCGA'
dcmbp = 'E:\\PhD\\Data'
aptc_cases = [case[:8] for case in listdir(join(aptc_path, 'images'))]
nii_cases = [case for case in listdir(nii_path) if case.endswith('.nii.gz') and not
             case.endswith('_pred.nii.gz') and case[:8] in aptc_cases]
aptc_di = pickle.load(open(join(aptc_path, 'data_info.pkl'), 'rb'))
nii_cases_di = [aptc_di[case[:8]] for case in nii_cases]

# %%
for case, di in zip(nii_cases, nii_cases_di):

    if di['dataset'] == 'Apollo':
        dcm_load_path = join(dcmbp, 'APOLLO2', 'Apollo_Beer', 'AP-'+di['pat_id'])
        im_dcms = [dcm for dcm in listdir(dcm_load_path) if not dcm.startswith('AP')]
        c_dcms = [dcm for dcm in listdir(dcm_load_path) if dcm.startswith('AP')]
    else:
        dcm_load_path = join(dcmbp, 'TCGA_Segmentations', 'Lucian', 'TCGA-'+di['pat_id'])
        im_dcms = [dcm for dcm in listdir(dcm_load_path) if not dcm.startswith('TCGA')]
        c_dcms = [dcm for dcm in listdir(dcm_load_path) if dcm.startswith('TCGA')]
    ds = pydicom.dcmread(join(dcm_load_path, im_dcms[0]))
    if ds.SliceThickness != 5.0:
        print('Skip! Use only scans with slice thickness 5.0')

    img = 