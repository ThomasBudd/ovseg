from ovseg.HPO.SHA import SHA
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("i", type=int)
args = parser.parse_args()

data_name = 'OV04'
preprocessed_name = 'pod_om_08_5'
i_process = args.i

parameter_names = [['training', 'opt_params', 'weight_decay']]

parameter_grids = [np.logspace(-4, -5, 10)]

target_metrics = ['dice_1', 'dice_9']
validation_set_name = 'BARTS'

patch_size = [32, 216, 216]
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

n_epochs_per_stage = [1000, 1000]
vfs_per_stage = [[5], [6,7]]
n_processes = 10

for mu in [0.99, 0.98, 0.97]:
    default_model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                                 z_to_xy_ratio=5.0/0.8,
                                                                 out_shape=out_shape,
                                                                 n_fg_classes=2,
                                                                 use_prg_trn=use_prg_trn)
    default_model_params['data']['trn_dl_params']['batch_size'] = 4
    default_model_params['data']['val_dl_params']['batch_size'] = 4
    default_model_params['training']['opt_params']['momentum'] = mu
    
    hpo_name = 'wd_'+str(mu)
    
    sha = SHA(data_name=data_name,
              preprocessed_name=preprocessed_name,
              i_process=i_process,
              parameter_names=parameter_names,
              parameter_grids=parameter_grids,
              target_metrics=target_metrics,
              validation_set_name=validation_set_name,
              default_model_params=default_model_params,
              n_epochs_per_stage=n_epochs_per_stage,
              vfs_per_stage=vfs_per_stage,
              hpo_name=hpo_name,
              n_processes=n_processes)
    sha.launch()