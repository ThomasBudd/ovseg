from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("raw_data", default='OV04', nargs='+',)
parser.add_argument("vf", type=int)
args = parser.parse_args()

vf = args.vf
raw_data = args.raw_data
data_name = '_'.join(sorted(raw_data))
preprocessed_name = 'lymph_nodes'

# hyper-paramteres for the model
wd = 1e-4
patch_size = [32, 256, 256]
use_prg_trn = True
out_shape = [[20, 160, 160],
             [24, 192, 192],
             [28, 224, 224],
             [32, 256, 256]]
larger_res_encoder = True

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=5.0/0.67,
                                                     out_shape=out_shape,
                                                     n_fg_classes=4,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder)
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = wd

# change the model name when using other hyper-paramters
model_name = 'clara_model'

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
ens.wait_until_all_folds_complete()
ens.eval_raw_dataset('BARTS')
ens.eval_raw_dataset('ApolloTCGA')