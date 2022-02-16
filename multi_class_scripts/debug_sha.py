from ovseg.HPO.SHA import SHA
from ovseg.model.model_parameters_segmentation import get_model_params_effUNet
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("i", type=int)
args = parser.parse_args()

data_name = 'OV04'
preprocessed_name = 'pod_067'
i_process = args.i
parameter_names = [['training', 'opt_params', 'momentum'], 
                   ['training', 'opt_params', 'weight_decay']]
parameter_grids = [[0.99, 0.98, 0.95, 0.9],
                   np.logspace(np.log10(3e-5), np.log10(3e-6), 4)]

target_metrics = ['dice_9']
validation_set_name = 'BARTS_small'
default_model_params = get_model_params_effUNet()
default_model_params['network']['filters'] = 8
default_model_params['training']['num_epochs'] = 50
default_model_params['data']['trn_dl_params']['num_workers'] = 5
default_model_params['data']['trn_dl_params']['epoch_len'] = 50
default_model_params['prediction']['mode'] = 'simple'

n_epochs_per_stage = [100, 200, 400, 400]
vfs_per_stage = [[5], [5], [5], [6, 7]]

hpo_name = 'DEBUG'
n_processes = 4

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