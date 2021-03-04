from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
import torch
import os
num_epochs = 1000

val_fold = 0
data_name = 'OV04'

architecture = 'post_processing_U_Net'
model_parameters = get_model_params_2d_reconstruction(architecture,
                                                      image_folder='images_HU_rescale',
                                                      projection_folder='fbp_projections_HU')
model_parameters['training']['num_epochs'] = num_epochs
model_parameters['training']['opt_params'] = {'lr': 10**-3, 'betas': (0.9, 0.999)}

model = Reconstruction2dSimModel(val_fold=val_fold,
                                 data_name=data_name,
                                 model_name=architecture,
                                 model_parameters=model_parameters)
model.training.train()
model.eval_validation_set(save_preds=True)
model.eval_training_set(save_preds=True)
torch.cuda.empty_cache()
