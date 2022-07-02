from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.SegmentationEnsembleV2 import SegmentationEnsembleV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
import argparse
import numpy as np
import os
from ovseg import OV_PREPROCESSED

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()
 
data_name = 'OV04'
preprocessed_name = 'pod_om_new'


wd = np.logspace(-5, -3.5,5)[args.exp]
# wd = 3*10**-5
lr_max = 0.02

if not os.path.exists(os.path.join(OV_PREPROCESSED, data_name, preprocessed_name)):
    lb_classes = [1,9]
    
    target_spacing=[[5.0, 0.8, 0.8]][args.exp]
    
    prep = SegmentationPreprocessingV2(apply_resizing=True, 
                                       apply_pooling=False, 
                                       apply_windowing=True,
                                       target_spacing=target_spacing,
                                       lb_classes=lb_classes)
    prep.plan_preprocessing_raw_data(data_name)

    prep.preprocess_raw_data(raw_data=data_name,
                             preprocessed_name=preprocessed_name)

patch_size = [32, 192, 192]
    
bs = 2    
model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     use_prg_trn=False,
                                                     larger_res_encoder=False,
                                                     n_fg_classes=2)

model_params['architecture'] = 'UNet'
model_params['network']['kernel_sizes'] = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
model_params['network']['in_channels'] = 1
model_params['network']['norm'] = 'inst'
del model_params['network']['block']
del model_params['network']['z_to_xy_ratio']
del model_params['network']['n_blocks_list']
del model_params['network']['stochdepth_rate']


for s in ['trn_dl_params', 'val_dl_params']:
    del model_params['data'][s]['store_coords_in_ram']
    del model_params['data'][s]['memmap']
model_params['training']['opt_params']['weight_decay'] = wd
model_params['training']['opt_params']['momentum'] = 0.99


model_name = f'ps_xy_{patch_size[1]}'
model_params['training']['lr_params']['lr_max'] = lr_max

model = SegmentationModelV2(val_fold=0,
                            data_name=data_name,
                            preprocessed_name=preprocessed_name, 
                            model_name=model_name,
                            model_parameters=model_params)
model.training.train()
model.eval_validation_set()