from ovseg.model.RegionexpertModel import RegionexpertModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

w = [0.01, 0.03, 0.1, 0.3][3]

for pref in ['pod', 'om']:
    p_name = pref+'_reg_expert_{}'.format(w)
    
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=[32, 216, 216],
                                                         z_to_xy_ratio=5.0/0.8,
                                                         use_prg_trn=False,
                                                         larger_res_encoder=False,
                                                         n_fg_classes=1)
    model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                              'dice_loss_weighted'],
                                              'loss_kwargs': [{'weight_bg': 1,
                                                               'n_fg_classes': 1},
                                                              {'eps': 1e-5,
                                                               'weight': 1}]}
    
    model_params['data']['folders'] = ['images', 'labels', 'regions']
    model_params['data']['keys'] = ['image', 'label', 'region']
    model_params['training']['batches_have_masks'] = True
    model_params['postprocessing'] = {'mask_with_reg': True}
    
    model = RegionexpertModel(val_fold=1+args.exp,
                              data_name='OV04',
                              preprocessed_name=p_name, 
                              model_name='regionexpert_ft',
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set(force_evaluation=True)
    model.eval_training_set(force_evaluation=True)
    model.eval_raw_dataset('BARTS', force_evaluation=True)
