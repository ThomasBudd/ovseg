from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
parser.add_argument("j", type=int)
args = parser.parse_args()

data_name = 'OV04'
preprocessed_name = 'pod_om_08_5'

wd = 1e-4

prev_model_name =  'bs4'

target_metrics = ['dice_1', 'dice_9']
validation_set_name = 'BARTS'
prev_preds = [data_name, preprocessed_name, prev_model_name,
              'cross_validation']

n_bias1 = [0, 0, 1][args.j]

n_bias2 = [1, 2, 1][args.j]

patch_size = [32, 216, 216]
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     n_fg_classes=2,
                                                     use_prg_trn=use_prg_trn,
                                                     out_shape=out_shape)

model_params['data']['use_double_bias'] = True
for s in ['trn_dl_params','val_dl_params']:
    model_params['data'][s]['batch_size'] = 4
    model_params['data'][s]['prev_preds'] = prev_preds
    model_params['data'][s]['n_bias1'] = n_bias1
    model_params['data'][s]['n_bias2'] = n_bias2
    model_params['data'][s]['lb_classes'] = [1,9]

model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = wd


model_name = 'CV_refine_{}_{}'.format(n_bias1, n_bias2)

#uncooment
model = SegmentationModel(val_fold=args.vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)

model.training.train()
model.eval_raw_data_npz('BARTS')
model.eval_raw_data_npz('ApolloTCGA')


ens = SegmentationEnsemble(val_fold=[5,6,7],
                            data_name=data_name,
                            model_name=model_name,
                            preprocessed_name=preprocessed_name)
ens.wait_until_all_folds_complete()
ens.eval_raw_dataset('ApolloTCGA')
ens.eval_raw_dataset('BARTS')