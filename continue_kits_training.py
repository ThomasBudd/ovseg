from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

vf_list = list(range(5))

w = [-1.5, -1.0, -0.5, 0.0, 0.5][args.exp]

patch_size = [32, 128, 128]

data_name = 'kits21'
p_name = 'kidney_tumour'

bs = 8
wd = 1e-4


model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=3.0/0.8,
                                                     use_prg_trn=False,
                                                     larger_res_encoder=False,
                                                     n_fg_classes=1)

model_params['architecture'] = 'UNet'
model_params['network']['kernel_sizes'] = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
model_params['network']['in_channels'] = 1
model_params['network']['norm'] = 'batch'
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
model_params['training']['opt_params']['momentum'] = 0.9
model_params['training']['loss_params']['loss_names'] = ['dice_loss_sigm_weighted',
                                                         'cross_entropy_exp_weight']
model_params['postprocessing'] = {'mask_with_reg': True}

w_list = [w]

model_params['training']['loss_params']['loss_kwargs'] = 2*[{'w_list':w_list}]

model_name = f'calibrated_{w:.2f}'

for vf in vf_list:
    
    model = SegmentationModelV2(val_fold=vf,
                                data_name=data_name,
                                preprocessed_name=p_name, 
                                model_name=model_name,
                                model_parameters=model_params)
    
    
    path_to_checkpoint = os.path.join(os.environ['OV_DATA_BASE'],
                                      'trained_models',
                                      data_name,
                                      p_name,
                                      model_name,
                                      f'fold_{vf}',
                                      'attribute_checkpoint.pkl')
    if os.path.exists(path_to_checkpoint):
        print('Previous checkpoint found and loaded')
    else:
        print('Loading pretrained checkpoint')
        path_to_checkpoint = os.path.join(os.environ['OV_DATA_BASE'],
                                          'trained_models',
                                          data_name,
                                          p_name,
                                          'stopped',
                                          f'fold_{vf}')
        model.training.load_last_checkpoint(path_to_checkpoint)
        model.training.loss_params = {'loss_names': ['dice_loss_sigm_weighted',
                                                     'cross_entropy_exp_weight'],
                                      'loss_kwargs': 2*[{'w_list':w_list}]}
        model.training.initialise_loss()
        model.training.save_checkpoint()
    model.training.train()
    model.eval_validation_set()