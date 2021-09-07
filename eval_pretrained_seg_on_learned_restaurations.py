import os
from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.data.Dataset import Dataset
import torch
import pickle

model_params = get_model_params_2d_segmentation()
del model_params['augmentation']['torch_params']['grayvalue']
model_params['network']['norm'] = 'inst'
model_name = '2d_num_epochs_ADAM_long'

# this is just creating a new blank segmentation model with random weights
# %% creat datasets
keys = ['image', 'label']
folders_full = ['restaurations_full', 'labels']
folders_quater = ['restaurations_quater', 'labels']
preprocessed_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_2d')
splits = pickle.load(open(os.path.join(preprocessed_path, 'splits.pkl'), 'rb'))
scans = splits[-1]['val']
ds_full = Dataset(scans, preprocessed_path, keys, folders_full)
ds_quater = Dataset(scans, preprocessed_path, keys, folders_quater)

for f in range(5, 8):
    seg_model = SegmentationModel(val_fold=f,
                                  data_name='OV04',
                                  model_name=model_name,
                                  preprocessed_name='pod_2d',
                                  dont_store_data_in_ram=True)
    # load pretrained weights
    path_to_weights = os.path.join(os.environ['OV_DATA_BASE'],
                                   'trained_models',
                                   'OV04',
                                   'pod_2d',
                                   '2d_num_epochs_ADAM_long',
                                   'fold_{}'.format(5),
                                   'network_weights_2000')

    seg_model.network.load_state_dict(torch.load(path_to_weights,
                                     map_location=torch.device('cuda')))
    
    seg_model.eval_ds(ds_full, 'validation_full', save_preds=False, merge_to_CV_results=True)
    seg_model.eval_ds(ds_quater, 'validation_quater', save_preds=False, merge_to_CV_results=True)
