from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.data.Dataset import low_res_ds_wrapper
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
parser.add_argument("vf", type=int)
args = parser.parse_args()
patch_size = [32, 216, 216]
p_name = 'pod_om_08_5'
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

all_weights = [[1.3, 1.1], [1.4, 1.2], [1.5, 1.3]]
weights = all_weights[args.exp]

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=2,
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_fg',
                                                          'dice_loss_vector_weighted'],
                                           'loss_kwargs': [{'weights_fg': weights},
                                                           {'weights': weights}]}
model_params['data']['val_dl_params']['n_fg_classes'] = 2
model_params['data']['trn_dl_params']['n_fg_classes'] = 2
model_params['data']['val_dl_params']['bias'] = 'cl_fg'
model_params['data']['trn_dl_params']['bias'] = 'cl_fg'

    
model = SegmentationModel(val_fold=args.vf,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name='U-Net4_weighted_{:.2f}_{:.2f}'.format(*weights),
                          model_parameters=model_params)

model.training.train()
model.eval_validation_set()
model.eval_raw_data_npz('BARTS_dcm')
model.eval_raw_data_npz('ApolloTCGA_dcm')

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name='OV04', 
                           model_name='U-Net4_weighted_{:.2f}_{:.2f}'.format(*weights),
                           preprocessed_name=p_name)
while not ens.all_folds_complete():
    sleep(20)

ens.eval_raw_dataset('BARTS_dcm')
ens.eval_raw_dataset('ApolloTCGA_dcm')