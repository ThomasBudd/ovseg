from ovseg.model.RestaurationModel import RestaurationModel
from ovseg.model.model_parameters_restauration import get_model_params_2d_restauration

model_params = get_model_params_2d_restauration()
model_params['training']['num_epochs'] = 2500
model_params['data']['trn_dl_params']['num_workers'] = 4
model_params['data']['val_dl_params']['num_workers'] = 4

model = RestaurationModel(val_fold=0, data_name='OV04', model_name='restauration_v3',
                          model_parameters=model_params, preprocessed_name='pod_2d')

model.training.train()
model.eval_validation_set(save_plots=True)
