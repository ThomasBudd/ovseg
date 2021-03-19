from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
import torch
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=True)
parser.add_argument("-p", '--pretrain_only', required=False, default=False, action="store_true")
args = parser.parse_args()

model_name = 'post_processing_U_Net'
if args.pretrain_only:
    num_epochs = 500
    lr_min = 0.5 * 10**-2
    model_name += '_pretrained'
else:
    num_epochs = 100
    lr_min = 0
    model_name += '_fully_trained'

val_fold = 0
data_name = args.data
if data_name == 'OV04':
    image_folder = 'images_HU_rescale'
    projection_folder = 'fbp_projections_HU'
    preprocessed_name = 'pod_default'
else:
    image_folder = 'images_HU_rescale'
    projection_folder = 'fbp_normal'
    preprocessed_name = 'default'

architecture = 'post_processing_U_Net'
model_parameters = get_model_params_2d_reconstruction(architecture,
                                                      image_folder=image_folder,
                                                      projection_folder=projection_folder)
model_parameters['training']['num_epochs'] = num_epochs
model_parameters['training']['opt_params'] = {'lr': 10**-2, 'betas': (0.9, 0.999)}
model_parameters['training']['lr_params']['lr_min'] = lr_min
model_parameters['data']['trn_dl_params']['batch_size'] = 12
model_parameters['data']['val_dl_params']['batch_size'] = 12

model = Reconstruction2dSimModel(val_fold=val_fold,
                                 data_name=data_name,
                                 model_name=model_name,
                                 model_parameters=model_parameters,
                                 preprocessed_name=preprocessed_name)
model.training.train()
model.eval_validation_set(save_preds=True)
model.eval_training_set(save_preds=True)
torch.cuda.empty_cache()
