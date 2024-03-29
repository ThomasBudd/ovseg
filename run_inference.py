import os
from ovseg.utils.io import save_nii_from_data_tpl, save_dcmrt_from_data_tpl, is_dcm_path
from ovseg.utils.label_utils import reduce_classes
from ovseg.utils.download_pretrained_utils import maybe_download_clara_models
from ovseg.model.ClaraWrappers import ClaraWrapperOvarian
from ovseg.data.Dataset import raw_Dataset
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("tst_data",
                    help='Name of the folder in $OV_DATA_BASE\raw_data that '
                    'contains the data to run the inference on.')
# add all the names of the labled training data sets as trn_data
parser.add_argument("--models", 
                    default=['pod_om', 'abdominal_lesions','lymph_nodes'], nargs='+',
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

args = parser.parse_args()

data_name = 'clara_models'
models = args.models
tst_data = args.tst_data

# if the pretrained models were not downloaded yet, we're doing it now here
maybe_download_clara_models()
path_to_clara_models = os.path.join(os.environ['OV_DATA_BASE'], 'clara_models')

# create the dataset to iterate over the scans
ds = raw_Dataset(tst_data)

# some variables we need for saving the predictions
pred_key = 'prediction'
lb_classes = []
if 'pod_om' in models:
    lb_classes.extend([1, 9])
if 'abdominal_lesions' in models:
    lb_classes.extend([1,2,3,5,6,7])
if 'lymph_nodes' in models:
    lb_classes.extend([13,14,15,17])

lb_classes = sorted(list(set((lb_classes))))

# %%

def save_prediction(data_tpl):
    filename = data_tpl['scan'] + '.nii.gz'
    out_file = data_tpl['scan'] + '.dcm'
    
    # all predictions are stored in the designated 'predictions' folder in the OV_DATA_BASE
    pred_folder = os.path.join(os.environ['OV_DATA_BASE'],
                               'predictions',
                               data_name,
                               '_'.join(sorted(models)),
                               tst_data)
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    # get storing info from the data_tpl
    # IMPORTANT: We will always store the prediction in original shape
    # not in preprocessed shape

    save_nii_from_data_tpl(data_tpl, os.path.join(pred_folder, filename), pred_key)
    
    if is_dcm_path(data_tpl['raw_image_file']):
        
        red_key = pred_key+'_dcm_export'
        data_tpl[red_key] = reduce_classes(data_tpl[pred_key], lb_classes)
        names = [str(lb) for lb in lb_classes]
        save_dcmrt_from_data_tpl(data_tpl, os.path.join(pred_folder, out_file),
                                 key=red_key, names=names)

# %% iterate over the dataset and save predictions

for data_tpl in tqdm(ds):
    
    # FLIP AND ROTATE IMAGE
    # the Clara wrapper flips and rotates the image as the monai reader
    # applies these when reading the data.
    # as we're reading the data with the ov_seg library, we have to flip
    # and rotate in the opposite direction first
    data_tpl['image'] = np.rot90(data_tpl['image'][::-1, :, ::-1], 1, (1,2))
    
    # compute prediciton
    pred = ClaraWrapperOvarian(data_tpl, 
                               models=models,
                               path_to_clara_models=path_to_clara_models)
    
    # save predictions move the z axis back to the front
    pred = np.moveaxis(pred, -1, 0)
    data_tpl[pred_key] = pred
    save_prediction(data_tpl)

print('Inference done!')
print('Saved predictions can be found here:')
pred_folder = os.path.join(os.environ['OV_DATA_BASE'],
                           'predictions',
                           data_name,
                           '_'.join(sorted(models)),
                           tst_data)
print(pred_folder)

