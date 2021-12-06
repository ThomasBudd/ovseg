from ovseg.HPO.GridSearch import GridSearch
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("i", type=int)
args = parser.parse_args()

data_name = 'OV04'
preprocessed_name = 'pod_om_08_5'
i_process = args.i

parameter_names = [['training', 'loss_params', 'loss_kwargs', 'delta'],
                   ['training', 'loss_params', 'loss_kwargs', 'gamma']]

parameter_grids = [np.linspace(0.1, 0.9, 9),
                   np.linspace(0.05, 0.5, 10)]

target_metrics = ['dice_1', 'dice_9']
validation_set_name = 'BARTS'

patch_size = [32, 216, 216]
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False


default_model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                             z_to_xy_ratio=5.0/0.8,
                                                             out_shape=out_shape,
                                                             n_fg_classes=2)
# default_model_params['architecture'] = 'unetresshuffleencoder'
default_model_params['training']['loss_params'] = {'loss_names': ['unifiedFocalLoss'],
                                                   'loss_kwargs': {'delta': 0.5,
                                                                   'gamma': 0,
                                                                   'eps': 1e-5}}

hpo = GridSearch(data_name,
                 preprocessed_name,
                 i_process,
                 parameter_names,
                 parameter_grids,
                 target_metrics,
                 validation_set_name,
                 default_model_params,
                 vfs=[5,6,7],
                 hpo_name='ufl',
                 n_processes=10)

hpo.launch()