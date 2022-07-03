from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.SegmentationEnsembleV2 import SegmentationEnsembleV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
import argparse
import numpy as np
import os
from ovseg import OV_PREPROCESSED

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()
 
data_name = 'OV04'
preprocessed_name = 'pod_om_cascade_new'


# wd = np.logspace(-5, -3.5,5)[args.exp]
wd = 1e-4
lr_max = 0.02

if not os.path.exists(os.path.join(OV_PREPROCESSED, data_name, preprocessed_name)):
    lb_classes = [1,9]
    
    target_spacing=[5.0, 0.8, 0.8]
    
    prep = SegmentationPreprocessingV2(apply_resizing=True, 
                                       apply_pooling=False, 
                                       apply_windowing=True,
                                       target_spacing=target_spacing,
                                       lb_classes=lb_classes,
                                       prev_stage_for_input={'data_name': data_name,
                                                             'preprocessed_name': 'pod_om_5_13',
                                                             'model_name': 'tuned_bs4'})
    prep.plan_preprocessing_raw_data(data_name)

    prep.preprocess_raw_data(raw_data=data_name,
                             preprocessed_name=preprocessed_name)

patch_size = [32, 216, 216]
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
z_to_xy_ratio = 5.0/0.8
larger_res_encoder = False
n_fg_classes = 2
    
bs = 4

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=z_to_xy_ratio,
                                                     out_shape=out_shape,
                                                     n_fg_classes=n_fg_classes,
                                                     use_prg_trn=True)
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = wd
model_params['network']['in_channels'] = 3

model_name = 'tuned_bs4'

model = SegmentationModelV2(val_fold=args.vf,
                            data_name=data_name,
                            preprocessed_name=preprocessed_name, 
                            model_name=model_name,
                            model_parameters=model_params)
model.training.train()
model.eval_validation_set()
model.eval_raw_data_npz('BARTS')
model.eval_raw_data_npz('ApolloTCGA')

ens = SegmentationEnsembleV2(val_fold=[0,1,2,3,4],
                             data_name=data_name,
                             model_name=model_name,
                             preprocessed_name=preprocessed_name)
ens.wait_until_all_folds_complete()
ens.eval_raw_dataset('BARTS')
ens.eval_raw_dataset('ApolloTCGA')
