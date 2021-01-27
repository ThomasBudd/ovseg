from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
import torch
import os
num_epochs = 500

val_fold = 0
data_name = 'OV04'

betas_list = [(0.9, 0.999), (0.99, 0.999)]

for i, betas in enumerate(betas_list):
    model_parameters = get_model_params_2d_reconstruction('LPD')
    model_parameters['training']['num_epochs'] = num_epochs
    model_parameters['training']['opt_params']['betas'] = betas
    model = Reconstruction2dSimModel(val_fold=val_fold,
                                     data_name=data_name,
                                     model_name='LPD{}'.format(i+2),
                                     model_parameters=model_parameters)
    model.training.train()
    model.eval_validation_set(save_preds=False)
    model.eval_training_set(save_preds=False)
    torch.cuda.empty_cache()
    for dl in [model.data.trn_dl, model.data.val_dl]:
        if dl.store_data_in_ram:
            del dl.data[:]
