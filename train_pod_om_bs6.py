from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
parser.add_argument("--only_last_fold", default=False, action='store_true')
args = parser.parse_args()

data_name = 'OV04'
preprocessed_name = 'pod_om_08_5'

# hyper-paramteres for the model
wd = 1e-4
patch_size = [32, 216, 216]
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False


momentum = [0.98, 0.98, 0.97][args.exp]
min_bias_samples = [1, 2, 2][args.exp]
norm = 'batch'
batch_size = 6



model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     out_shape=out_shape,
                                                     n_fg_classes=2,
                                                     use_prg_trn=use_prg_trn)
model_params['architecture'] = 'unetresencoderv2'
model_params['network']['norm'] = norm
del model_params['network']['block']
del model_params['network']['stochdepth_rate']
model_params['data']['trn_dl_params']['batch_size'] = batch_size
model_params['data']['val_dl_params']['batch_size'] = batch_size
model_params['data']['trn_dl_params']['min_bias_samples'] = min_bias_samples
model_params['data']['val_dl_params']['min_bias_samples'] = min_bias_samples
model_params['training']['opt_params']['momentum'] = momentum
model_params['training']['opt_params']['weight_decay'] = wd

# change the model name when using other hyper-paramters
model_name = f'bs{batch_size}_{norm}_{momentum}_{min_bias_samples}'

if args.only_last_fold:

    model = SegmentationModel(val_fold=9,
                              data_name=data_name,
                              model_name=model_name,
                              preprocessed_name=preprocessed_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_raw_data_npz('BARTS')
    model.eval_raw_data_npz('ApolloTCGA')
    model.clean()

else:
    
    for vf in [5,6,7]:
        model = SegmentationModel(val_fold=vf,
                                  data_name=data_name,
                                  model_name=model_name,
                                  preprocessed_name=preprocessed_name,
                                  model_parameters=model_params)
        model.training.train()
        model.eval_raw_data_npz('BARTS')
        model.eval_raw_data_npz('ApolloTCGA')
        model.clean()
        
    ens = SegmentationEnsemble(val_fold=[5,6,7],
                               data_name=data_name,
                               model_name=model_name,
                               preprocessed_name=preprocessed_name)
    ens.wait_until_all_folds_complete()
    ens.eval_raw_dataset('BARTS')
    ens.eval_raw_dataset('ApolloTCGA')
    for vf in [8, 9]:
        model = SegmentationModel(val_fold=vf,
                                  data_name=data_name,
                                  model_name=model_name,
                                  preprocessed_name=preprocessed_name,
                                  model_parameters=model_params)
        model.training.train()
        model.eval_raw_data_npz('BARTS')
        model.eval_raw_data_npz('ApolloTCGA')
        model.clean()
    ens = SegmentationEnsemble(val_fold=[5,6,7, 8, 9],
                               data_name=data_name,
                               model_name=model_name,
                               preprocessed_name=preprocessed_name)
    ens.wait_until_all_folds_complete()
    ens.eval_raw_dataset('BARTS')
    ens.eval_raw_dataset('ApolloTCGA')