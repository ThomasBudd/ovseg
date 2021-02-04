from ovseg.preprocesing.SegmentationPreprocessing import SegmentationPreprocessing
import os
from ovseg.utils.io import read_nii_files
import numpy as np


preprocessing = SegmentationPreprocessing()
prep = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_default')
preprocessing.load_preprocessing_parameters(prep)

model_names = ['recon_LPD_full_HU', 'recon_fbp_convs_full_HU', 'reconstruction_network_fbp_convs']

for model_name in model_names:
    print(model_name)
    path_to_fold = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', 'OV04', model_name,
                                'fold_0')
    image_folder = os.path.join(prep, model_name)
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

    for folder in ['training', 'validation']:
        print(folder)
        pred_path = os.path.join(path_to_fold, folder, 'predictions')
        for nii_file in os.listdir(pred_path):
            volume, spacing = read_nii_files(os.path.join(pred_path, nii_file))
            if len(volume.shape) == 4:
                volume = volume[0]
            volume_prep = preprocessing.preproces_volume(volume, spacing)
            np.save(os.path.join(image_folder, nii_file[:8]+'.npy'),
                    volume_prep.astype(np.float32))
