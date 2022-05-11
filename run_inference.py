import os
from ovseg.utils.io import save_nii_from_data_tpl, save_dcmrt_from_data_tpl, is_dcm_path
from ovseg.utils.label_utils import reduce_classes
from ovseg.model.ClaraWrappers import ClaraWrapperOvarian
from ovseg.data.Dataset import raw_Dataset
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("tst_data")
# add all the names of the labled training data sets as trn_data
parser.add_argument("--trn_data", default=['OV04', 'BARTS', 'ApolloTCGA'], nargs='+')
parser.add_argument("--models", default=['pod_om', 'abdominal_lesions','lymph_nodes'], nargs='+')

args = parser.parse_args()

trn_data = args.trn_data
data_name = '_'.join(sorted(trn_data))
models = args.models
tst_data = args.tst_data

# change the model name when using other hyper-paramters
model_name = 'clara_model'
# if you store the weights and model parameters somewhere else, please change
# this path
path_to_clara_models = os.path.join(os.environ['OV_DATA_BASE'], 'clara_models')

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

