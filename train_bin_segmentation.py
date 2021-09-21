from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

use_prg_trn = (args.exp % 2) == 1

p_name = 'bin_seg'

if args.exp < 2:
    patch_size = [28, 216, 216]
    model_name = 'U-Net4'
    if use_prg_trn:
        model_name += '_prg_lrn'
    larger_res_encoder = False
elif args.exp < 4:
    patch_size = [28, 216, 216]
    model_name = 'U-Net5'
    if use_prg_trn:
        model_name += '_prg_lrn'
    larger_res_encoder = True
else:
    patch_size = [40, 320, 320]
    model_name = 'U-Net5_large_patches'
    if use_prg_trn:
        model_name += '_prg_lrn'
    larger_res_encoder = True
    


model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/0.67,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=1)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                          'dice_loss_weighted'],
                                          'loss_kwargs': [{'weight_bg': 1,
                                                           'n_fg_classes': 1},
                                                          {'eps': 1e-5,
                                                           'weight': 1}]}

model = SegmentationModel(val_fold=0,
                          data_name='OV04',
                          preprocessed_name=p_name, 
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()
model.eval_raw_dataset('BARTS')
