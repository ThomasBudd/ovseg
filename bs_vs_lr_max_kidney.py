from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from ovseg import OV_PREPROCESSED

data_name = 'kits21'
preprocessed_name = 'kidney_tumour'

if not os.path.exists(os.path.join(OV_PREPROCESSED, data_name, preprocessed_name)):
    
    prev_stage_for_mask = {'data_name': 'kits21',
                           'preprocessed_name': 'kidney_full_refine_refine',
                           'model_name': 'refine_model'}
    
    prep = SegmentationPreprocessingV2(apply_resizing=True,
                                       apply_pooling=False,
                                       apply_windowing=True,
                                       lb_classes=[2],
                                       save_only_fg_scans=False,
                                       scaling=[60.1791, 63.9565],
                                       target_spacing=[3.0, 0.78, 0.78],
                                       window=[-52, 286],
                                       prev_stage_for_mask=prev_stage_for_mask)
    
    # prep.plan_preprocessing_raw_data('kits21')
    prep.preprocess_raw_data(data_name, preprocessed_name)



patch_size = [32, 128, 128]

sizes = 16*np.round(patch_size[2] / np.arange(4,0,-1)**(1/3) / 16)
out_shape = [ [int(s)//4,  int(s), int(s)] for s in sizes]
print(out_shape)

bs = 2
wd = 1e-4

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=3.0/0.8,
                                                     use_prg_trn=True,
                                                     out_shape=out_shape,
                                                     larger_res_encoder=False,
                                                     n_fg_classes=1)

model_params['architecture'] = 'UNet'
model_params['network']['kernel_sizes'] = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
model_params['network']['in_channels'] = 1
model_params['network']['norm'] = 'inst'
del model_params['network']['block']
del model_params['network']['z_to_xy_ratio']
del model_params['network']['n_blocks_list']
del model_params['network']['stochdepth_rate']

model_params['data']['folders'] = ['images', 'labels', 'masks']
model_params['data']['keys'] = ['image', 'label', 'mask']

for s in ['trn_dl_params', 'val_dl_params']:
    model_params['data'][s]['batch_size'] = bs
    model_params['data'][s]['min_biased_samples'] = bs//3
    model_params['data'][s]['num_workers'] = 14
    del model_params['data'][s]['store_coords_in_ram']
    del model_params['data'][s]['memmap']
model_params['training']['batches_have_masks'] = True
model_params['training']['opt_params']['weight_decay'] = wd
model_params['training']['opt_params']['momentum'] = 0.99
model_params['training']['stop_after_epochs'] = [100]

model_params['postprocessing'] = {'mask_with_reg': True}

for lr_max in [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]:

    model_name = f'bs_{bs}_lr_max_{lr_max:.2f}'
    model_params['training']['lr_params']['lr_max'] = lr_max
    
    model = SegmentationModelV2(val_fold=0,
                                data_name=data_name,
                                preprocessed_name=preprocessed_name, 
                                model_name=model_name,
                                model_parameters=model_params)
    if model.training.epochs_done < 100:
        model.training.train()
