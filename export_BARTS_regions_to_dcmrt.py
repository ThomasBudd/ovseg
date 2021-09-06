import nibabel as nib
import pickle
import os
from rt_utils import RTStructBuilder
import numpy as np

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 'OV04', 'multiclass_reg',
                     'regfinding_0.1', 'BARTS_fold_0')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'BARTS_dcm')
with open(os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'data_info.pkl'), 'rb') as file:
    data_info = pickle.load(file)

# %%
for case in os.listdir(predp):
    if not case.startswith('case'):
        continue
    i = int(case[5:8])
    pred = nib.load(os.path.join(predp, 'case_{}.nii.gz'.format(i))).get_fdata()
    scan_path = data_info[str(i)]['scan']
    pat_id = os.path.basename(scan_path)
    
    dcm_rts = [f for f in os.listdir(scan_path) if f.startswith('ID')]
    
    if len(dcm_rts) ==0:
        continue
    
    dcms = [f for f in os.listdir(scan_path) if not f.startswith('ID')]
    
    if len(dcms) != pred.shape[-1]:
        print('Skipping, found slice thickness other than 5')
        continue
    dcm_rt = dcm_rts[0]
    
    rtstruct = RTStructBuilder.create_from(dicom_series_path=scan_path,
                                           rt_struct_path=os.path.join(scan_path, dcm_rt))
    
    classes = np.unique(pred)
    for c in classes:
        if c == 0:
            continue
        rtstruct.add_roi(mask=pred[..., ::-1] == c, name='{}-region'.format(int(c)))
    
    rtstruct.save(os.path.join(predp, '{}_regions.dcm'.format(pat_id)))
