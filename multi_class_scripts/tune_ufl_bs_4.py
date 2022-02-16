from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("i", type=int)
args = parser.parse_args()

data_name = 'OV04'
preprocessed_name = 'pod_om_08_5'

wd = 1e-4

patch_size = [32, 216, 216]
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                             z_to_xy_ratio=5.0/0.8,
                                                             out_shape=out_shape,
                                                             n_fg_classes=2,
                                                             use_prg_trn=use_prg_trn)
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = wd
model_params['training']['loss_params'] = {'loss_names': ['unifiedFocalLoss'],
                                           'loss_kwargs': {'delta': (args.i+1)/10,
                                                           'gamma': 0,
                                                           'eps': 1e-5}}

model_name = 'tune_ufl_1_'+str(args.i)

for vf in [5,6,7]:
    model = SegmentationModel(val_fold=vf,
                              data_name=data_name,
                              model_name=model_name,
                              preprocessed_name=preprocessed_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_raw_dataset('BARTS')
    model.eval_raw_dataset('ApolloTCGA')

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=preprocessed_name)
ens.eval_raw_dataset('BARTS')
ens.eval_raw_dataset('ApolloTCGA')