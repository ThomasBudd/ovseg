from ovseg.model.RestaurationModel import RestaurationModel
from ovseg.model.model_parameters_restauration import get_model_params_2d_restauration

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

fbp_folder = ['fbps_full', 'fbps_half', 'fbps_quater', 'fbps_eights', 'fbps_16', 'fbps_32'][args.exp]


model_params = get_model_params_2d_restauration()
model_params['training']['num_epochs'] = 2500
model_params['training']['compute_val_psnr_everk_k_epochs'] = 100
model_params['data']['folders'] = ['images_restauration', fbp_folder]
model = RestaurationModel(val_fold=0,
                          data_name='OV04',
                          model_name='restauration_debug',
                          model_parameters=model_params,
                          preprocessed_name='pod_2d')

model.training.train()
model.eval_validation_set(save_plots=True)
