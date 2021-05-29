from ovseg.model.RestaurationModel import RestaurationModel
from ovseg.model.model_parameters_restauration import get_model_params_2d_restauration

model_params = get_model_params_2d_restauration()
model_params['training']['num_epochs'] = 2500


for model_name, b in zip(['restauration_v1', 'restauration_v2'], [False, True]):
    model_params['network']['double_channels_every_second_block'] = b
    model = RestaurationModel(val_fold=0, data_name='OV04', model_name=model_name,
                              model_parameters=model_params, preprocessed_name='pod_2d')
    
    model.training.train()
    model.eval_validation_set(save_plots=True)
