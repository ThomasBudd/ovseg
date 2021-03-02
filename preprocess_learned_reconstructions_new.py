from ovseg.preprocesing.SegmentationPreprocessing import SegmentationPreprocessing
import os
from ovseg.utils.io import read_nii
import numpy as np
from tqdm import tqdm

model_name = 'recon_fbp_convs_high_0_long'
data_name = 'OV04'
preprocessed_name = 'pod_default'

preprocessing = SegmentationPreprocessing()
prep = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', data_name, preprocessed_name)
preprocessing.load_preprocessing_parameters(prep)

path_to_predictions = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', data_name,
                                   model_name)
path_to_prep_recons = os.path.join(prep, model_name)
if not os.path.exists(path_to_prep_recons):
    os.mkdir(path_to_prep_recons)

for f in ['training_0', 'validation_0']:
    print(f)
    print()
    cases = os.listdir(os.path.join(path_to_predictions, f))
    for case in tqdm(cases):
        im, spacing = read_nii(os.path.join(path_to_predictions, f, case))
        im_prep = preprocessing.preprocess_volume(im, spacing)
        np.save(os.path.join(path_to_prep_recons, case.split('.')[0]), im_prep)
