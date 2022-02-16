from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.data.Dataset import low_res_ds_wrapper
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

p_name = 'bin_seg'
patch_size = [40, 320, 320]
use_prg_trn = True
out_shape = [[24, 192, 192], #4
             [28, 224, 224], #2.3
             [36, 288, 288], #1.26
             [40, 320, 320]] #1
larger_res_encoder = True

N_PROC = 4

M_list = np.arange(5, 21)[args.exp::N_PROC]
scale = (np.array(out_shape[0]) / np.array(out_shape[-1])).tolist()

BARTS_low_res_ds = low_res_ds_wrapper('BARTS', scale)

for M in M_list:
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                         z_to_xy_ratio=5.0/0.67,
                                                         use_prg_trn=use_prg_trn,
                                                         larger_res_encoder=larger_res_encoder,
                                                         n_fg_classes=1,
                                                         out_shape=out_shape)
    model_params['training']['prg_trn_aug_params'] = {'M': [M/4, M],
                                                      'out_shape': out_shape}
    model_params['training']['prg_trn_resize_on_the_fly'] = False
    model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                              'dice_loss_weighted'],
                                              'loss_kwargs': [{'weight_bg': 1,
                                                               'n_fg_classes': 1},
                                                              {'eps': 1e-5,
                                                               'weight': 1}]}
    model_params['training']['stop_after_epochs'] = [250, 500]
    del model_params['augmentation']['torch_params']['grayvalue']
    model_params['augmentation']['torch_params']['myRandAugment'] = {'M': M, 'P':0.15}
    
    model = SegmentationModel(val_fold=0,
                              data_name='OV04',
                              preprocessed_name=p_name, 
                              model_name='U-Net5_M_{}'.format(M),
                              model_parameters=model_params)

    if model.training.epochs_done < 250:
        model.training.train()
        model.eval_ds(BARTS_low_res_ds, 'BARTS_250', save_preds=False)
