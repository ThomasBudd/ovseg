from ovseg.model.RestaurationModel import RestaurationModel
from ovseg.model.model_parameters_restauration import get_model_params_2d_restauration

model_params = get_model_params_2d_restauration()
model_params['training']['num_epochs'] = 100
model_params['training']['compute_val_psnr_everk_k_epochs'] = 10
model_params['data']['trn_dl_params']['batch_size'] = 2
model_params['data']['val_dl_params']['batch_size'] = 2
model_params['network']['filters'] = 8
model_params['data']['folders'] = ['images_restauration_full', 'fbps_full']
model = RestaurationModel(val_fold=0, data_name='OV04', model_name='restauration_debug',
                          model_parameters=model_params, preprocessed_name='pod_2d')

model.training.train()
model.eval_validation_set(save_plots=True)
