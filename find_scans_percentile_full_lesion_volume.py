import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from skimage.measure import label
import pickle
from ovseg.utils.io import read_dcms, save_dcmrt_from_data_tpl

lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
                   'OV04', 'labels')
# %%
vols = []
for scan in tqdm(os.listdir(lbp)):
    
    img = nib.load(os.path.join(lbp, scan))
    vv = np.prod(img.header['pixdim'][1:4])/1000
    lb = (img.get_fdata() > 0).astype(float)
    vols.append(np.sum(lb) * vv)

vols = np.array(vols)
# %%
prec = np.percentile(vols, [25,50,75,90])
cases = []
dcmp =  os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
                   'OV04_dcm')
path_list = []
for p in prec:
    cases.append(np.argmin(np.abs(vols - p)))

di = pickle.load(open(os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
                   'OV04', 'data_info.pkl'), 'rb'))

for case in cases:
    scan = 'case_{:03d}.nii.gz'.format(case)
    img = nib.load(os.path.join(lbp, scan))
    vv = np.prod(img.header['pixdim'][1:4])/1000
    lb = (img.get_fdata() > 0).astype(float)
    n_comps = np.max(label(lb))
    info = di[scan.split('.')[0]]
    print(info['pat_id'], info['timepoint'])
    print(scan + ' {:.2f}, {}'.format(vv*np.sum(lb), n_comps))
    
    patp = os.path.join(dcmp, info['pat_id'])
    
    if info['timepoint'] == 'BL':
        path_list.append(os.path.join(patp, os.listdir(patp)[0]))
    else:
        path_list.append(os.path.join(patp, os.listdir(patp)[1]))
# %%
for path in path_list:
    if os.path.exists(os.path.join(path, 'binary_segmentation.dcm')):
        os.remove(os.path.join(path, 'binary_segmentation.dcm'))

for path in path_list:
    data_tpl = read_dcms(path)
    data_tpl['label'] = (data_tpl['label'] > 0).astype(float)
    out_file = os.path.join(path, 'binary_segmentation.dcm')
    save_dcmrt_from_data_tpl(data_tpl, out_file, key='label')