from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("w", type=int)
parser.add_argument("m", type=int)
args = parser.parse_args()

patch_size = [32, 216, 216]
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

data_name = 'OV04'
p_name = 'pod_om_08_5'

model_name = 'bs4_wd_mu_{}_{}'.format(args.m, args.w)

momentum = [0.99, 0.97][args.m]
weight_decay = np.logspace(np.log10(5e-5), np.log10(5e-6), 5)[args.w]

model_names = ['U-Net4', 'effU-Net4']
architctures = ['unetresencoder', 'unetresshuffleencoder']


model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5/0.8,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=2,
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                          'dice_loss']}
model_params['training']['opt_params']['momentum'] = momentum
model_params['training']['opt_params']['weight_decay'] = weight_decay
    
for vf in [5,6,7]:
    
    model = SegmentationModel(val_fold=vf,
                              data_name=data_name,
                              preprocessed_name=p_name, 
                              model_name=model_name,
                              model_parameters=model_params)
    model.training.train()

ens = SegmentationEnsemble(val_fold=list(range(5,8)),
                           data_name=data_name,
                           preprocessed_name=p_name, 
                           model_name=model_name)

ens.eval_raw_dataset('BARTS')
