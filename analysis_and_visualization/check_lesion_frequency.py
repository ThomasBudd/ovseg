import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import pydicom
from ovseg.utils.io import _is_roi_dcm_ds

for data in ['OV04', 'BARTS', 'ApolloTCGA']:
    lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', data, 'labels')
    cases = os.listdir(lbp)
    
    classes_list = []
    classes_vec = np.zeros(22)
    
    for case in tqdm(cases):
        seg = nib.load(os.path.join(lbp, case)).get_fdata()
        classes = np.unique(seg)
        for c in classes:
            classes_vec[int(c)] += 1
        classes_list.append(classes)
    
    classes_vec /= len(cases)
    
    for i, c in enumerate(classes_vec):
        print(i, ': ', c * 100)

# %%

les_names = []
dp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04_dcm')
for pat in tqdm(os.listdir(dp)):
    for scan in os.listdir(os.path.join(dp, pat)):
        if scan.endswith('.txt'):
            continue
        scanp = os.path.join(dp, pat, scan)
        dcms = [os.path.join(scanp, dcm) for dcm in os.listdir(scanp) if dcm.endswith('dcm')]
        ds1 = pydicom.dcmread(dcms[0])
        ds2 = pydicom.dcmread(dcms[1])
        ds = None
        if _is_roi_dcm_ds(ds1):
            ds = ds1
        elif _is_roi_dcm_ds(ds2):
            ds = ds2
        if ds is None:
            for dcm in dcms:
                ds = pydicom.dcmread(dcm)
                if _is_roi_dcm_ds(ds):
                    break
        if hasattr(ds, 'StructureSetROISequence'):
            names_found = [s.ROIName.lower() for s in ds.StructureSetROISequence]
            les_names.extend(names_found)
            les_names = np.unique(les_names).tolist()

for name in sorted(les_names):
    print(name)
