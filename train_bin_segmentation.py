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
        out_shape = [[16, 136, 136], #4
                     [20, 160, 160], #2.3
                     [24, 192, 192], #1.26
                     [28, 216, 216]] #1
    else:
        out_shape = None
    larger_res_encoder = False
elif args.exp < 4:
    patch_size = [28, 216, 216]
    model_name = 'U-Net5'
    if use_prg_trn:
        model_name += '_prg_lrn'
        out_shape = [[16, 136, 136], #4
                     [20, 160, 160], #2.3
                     [24, 192, 192], #1.26
                     [28, 216, 216]] #1
    else:
        out_shape = None
    larger_res_encoder = True
elif args.exp < 6:
    patch_size = [40, 320, 320]
    model_name = 'U-Net5_large_patches'
    if use_prg_trn:
        model_name += '_prg_lrn'
        out_shape = [[24, 192, 192], #4
                     [28, 224, 224], #2.3
                     [36, 288, 288], #1.26
                     [40, 320, 320]] #1
    else:
        out_shape = None
    larger_res_encoder = True
else:
    use_prg_trn = True
    model_name = 'U-Net5_very_large_patches_prg_lrn'
    patch_size = [48, 384, 384]
    larger_res_encoder = True
    out_shape = [[28, 244, 244],
                 [32, 256, 256],
                 [40, 320, 320],
                 [48, 384, 384]]


model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/0.67,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=1,
                                                     out_shape=out_shape)
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
