import os
from ovseg.utils.io import read_nii, save_nii
from ovseg.utils.download_pretrained_utils import maybe_download_clara_models
from ovseg.model.InferenceWrapper import InferenceWrapper
import argparse


def is_nii_file(path_to_file):
    return path_to_file.endswith('.nii') or path_to_file.endswith('.nii.gz')

    
def run_inference(path_to_data,
                  models=['pod_om'],
                  fast=False):

    if is_nii_file(path_to_data):
        path_to_data, nii_file = os.path.split(path_to_data)
        nii_files = [nii_file]
    else:
        nii_files = [f for f in os.listdir(path_to_data) if is_nii_file(f)]
        if len(nii_files) == 0:
            raise FileNotFoundError(f"No nifti images were found at {path_to_data}")
    
    # if the pretrained models were not downloaded yet, we're doing it now here
    maybe_download_clara_models()
    
    # some variables we need for saving the predictions
    pred_folder_name = "ovseg_predictions"
    if 'pod_om' in models:
        pred_folder_name += "_pod_om"
    if 'abdominal_lesions' in models:
        pred_folder_name += "_abdominal_lesions"
    if 'lymph_nodes' in models:
        pred_folder_name += "_lymph_nodes"
    
    # iterate over the dataset and save predictions
    out_folder = os.path.join(path_to_data, pred_folder_name)
    os.makedirs(out_folder, exist_ok=True)
    
    for i, nii_file in enumerate(nii_files):
        
        print(f"Evaluate image {i} out of {len(nii_files)}")
        
        im, sp = read_nii(os.path.join(path_to_data, nii_file))
        
        pred = InferenceWrapper(im, sp, models, fast=fast)
        
        save_nii(pred, os.path.join(out_folder, nii_file), os.path.join(path_to_data, nii_file))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_data",
                        help='Either (i) path to a single nifti file like PATH/TO/IMAGE.nii(.gz),\n '
                        '(ii) path to a folder containing mulitple nifti files.')
    # add all the names of the labled training data sets as trn_data
    parser.add_argument("--models", 
                        default=['pod_om'], nargs='+',
                        help='Name(s) of models used during inference. Options are '
                        'the following.\n'
                        '(i) pod_om: model for main disease sites in the pelvis/ovaries'
                        ' and the omentum. The two sites are encoded as 9 and 1.\n'
                        '(ii) abdominal_lesions: model for various lesions between '
                        'the pelvis and diaphram. The model considers lesions in the '
                        'omentum (1), right upper quadrant (2), left upper quadrant (3), '
                        'mesenterium (5), left paracolic gutter (6) and right '
                        'paracolic gutter (7).\n'
                        '(iii) lymph_nodes: segments disease in the lymph nodes '
                        'namely infrarenal lymph nodes (13), suprarenal lymph nodes '
                        '(14), supradiaphragmatic lymph nodes (15) and inguinal '
                        'lymph nodes (17).\n'
                        'Any combination of the three are viable options.')
    parser.add_argument("--fast", action='store_true',
                        default=False,
                        help='Increases inference speed by disabling dynamic z spacing, '
                        'model ensembling and test-time augmentations.')
    
    args = parser.parse_args()
    
    path_to_data = args.path_to_data
    models = args.models
    fast = args.fast
    
    run_inference(path_to_data, models, fast)