from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.data.utils import split_scans_by_patient_id
from ovseg.utils import io
import os
import numpy as np

# first we obtain parameters from the OV04 dataset and preprocess it.
raw_data = 'OV04'
preprocessed_name = 'om_default'

# create preprocessing object and determine that we want to segment only class 9
preprocessing = SegmentationPreprocessing(use_only_classes=[1])
preprocessing.plan_preprocessing_raw_data(raw_data)
preprocessing.preprocess_raw_data(raw_data, preprocessed_name=preprocessed_name)

# the path where the preprocessed data lies
preprocessed_path = os.path.join(os.environ['OV_DATA_BASE'],
                                 'preprocessed',
                                 'OV04',
                                 preprocessed_name)

# %% split the OV04 data in a 4 fold CV
OV04_scans = os.listdir(os.path.join(preprocessed_path, 'images'))

# we obtain the patient ids from the fingerprints. We need them to make a split such that
# the baseline scan of a patient is always in the same fold as the follow-up scan
patient_ids = {}
for scan in OV04_scans:
    fngprnt = np.load(os.path.join(preprocessed_path, 'fingerprints', scan),
                      allow_pickle=True).item()
    patient_ids[scan] = fngprnt['dataset'] + '_' + fngprnt['pat_id']

# this is a typical 4 fold CV split of the OV04 data
splits = split_scans_by_patient_id(OV04_scans,
                                   patient_ids,
                                   n_folds=4,
                                   fixed_shuffle=True)

# %% preprocess other data
raw_data = ['BARTS', 'ApolloTCGA']
preprocessing.preprocess_raw_data(raw_data,
                                  preprocessed_name=preprocessed_name,
                                  data_name='OV04')
all_scans = os.listdir(os.path.join(preprocessed_path, 'images'))
BAT_scans = [scan for scan in all_scans if scan not in OV04_scans]

# %% add the 0th fold, all OV04 scans for training and the rest for validation
splits = [{'train': OV04_scans, 'val': BAT_scans}] + splits
path_to_splits = os.path.join(preprocessed_path, 'splits.pkl')
io.save_pkl(splits, path_to_splits)
