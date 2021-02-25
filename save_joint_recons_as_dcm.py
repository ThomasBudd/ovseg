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
        cdcm = [dcm for dcm in listdir(dcm_load_path) if dcm.startswith('AP')][0]
    else:
        dcm_load_path = join(dcmbp, 'TCGA_Segmentations', 'Lucian', 'TCGA-'+di['pat_id'])
        im_dcms = [dcm for dcm in listdir(dcm_load_path) if not dcm.startswith('TCGA')]
        cdcm = [dcm for dcm in listdir(dcm_load_path) if dcm.startswith('TCGA')][0]
    ds = pydicom.dcmread(join(dcm_load_path, im_dcms[0]))
    if ds.SliceThickness != 5.0:
        print('Skip! Use only scans with slice thickness 5.0')
        continue
    # first maybe create the folder
    dcm_save_path = join(nii_path, di['dataset']+'-'+di['pat_id'])
    if not exists(dcm_save_path):
        mkdir(dcm_save_path)
    for folder in ['vendor', 'us']:
        if not exists(join(dcm_save_path, folder)):
            mkdir(join(dcm_save_path, folder))

    # now copy the vendor data over
    for dcm in im_dcms + [cdcm]:
        if not exists(join(dcm_save_path, 'vendor', dcm)):
            copyfile(join(dcm_load_path, dcm),
                     join(dcm_save_path, 'vendor', dcm))
    if not exists(join(dcm_save_path, 'us', cdcm)):
        copyfile(join(dcm_load_path, cdcm),
                 join(dcm_save_path, 'us', cdcm))

    # now the tricky part: put our reconstructions
    img = nib.load(join(nii_path, case)).get_fdata() * 400 + 50
    ds_list = [pydicom.dcmread(join(dcm_load_path, dcm)) for dcm in im_dcms]
    for z, (dcm, ds) in enumerate(zip(im_dcms, ds_list)):
        im = (img[..., z] - float(ds.RescaleIntercept)) / float(ds.RescaleSlope)
        ds.PixelData = im.astype(int).tobytes()
        ds.save_as(join(dcm_save_path, 'us', dcm))
