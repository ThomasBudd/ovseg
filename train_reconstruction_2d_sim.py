from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
import torch
num_epochs = 10

val_fold = 0
data_name = 'OV04'

for architecture in ['reconstruction_network_fbp_convs', 'LPD']:
    model_parameters = get_model_params_2d_reconstruction(architecture)
    model_parameters['training']['num_epochs'] = num_epochs

    model = Reconstruction2dSimModel(val_fold=val_fold,
                                     data_name=data_name,
                                     model_name=architecture,
                                     model_parameters=model_parameters)
    model.training.train()
    model.eval_validation_set(save_preds=False)
    model.eval_training_set(save_preds=False)
    torch.cuda.empty_cache()
