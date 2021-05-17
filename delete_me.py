import os
import pickle
from ovseg.data.utils import split_scans_by_patient_id
import numpy as np

splitp = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_half', 'splits.pkl')
fpp = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_half', 'fingerprints')
splits = pickle.load(open(splitp, 'rb'))

scans = splits[-1]['train']
patient_ids = {}
for scan in scans:
    fp = np.load(os.path.join(fpp, scan), allow_pickle=True).item()
    patient_ids[scan] = fp['pat_id']
split_3CV = split_scans_by_patient_id(scans, patient_ids, n_folds=3)

splits_new = splits[:5] + split_3CV + splits[-1:]
pickle.dump(splits_new, open(splitp, 'wb'))