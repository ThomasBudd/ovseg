from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data')
parser.add_argument("-p", "--pretrain_only", required=False, default=False, action='store_true')
parser.add_argument('--use_windowed_simulations', required=False, default=False, action='store_true')

args = parser.parse_args()

model_name = 'recon_fbp_convs'

if args.use_windowed_simulations:
    model_name += '_win'
    projf = 'projections_normal_win'
    imf = 'images_HU_win_rescale'
else:
    projf = 'projections_normal'
    imf = 'images_HU_rescale'

if args.pretrain_only:
    model_name += '_pretrained'
    num_epochs = 500
else:
    model_name += '_fully_trained'
    num_epochs = 1000

val_fold = 0
data_name = args.data

if data_name == 'OV04':
    preprocessed_name = 'pod_default'
else:
    preprocessed_name = 'default'


model_parameters = get_model_params_2d_reconstruction('reconstruction_network_fbp_convs',
                                                      imf,
                                                      projf)
model_parameters['training']['num_epochs'] = num_epochs

model = Reconstruction2dSimModel(val_fold=val_fold,
                                 data_name=data_name,
                                 model_name=model_name,
                                 model_parameters=model_parameters,
                                 preprocessed_name=preprocessed_name)
model.training.train()
model.eval_validation_set(save_preds=True)
model.eval_training_set(save_preds=True)
torch.cuda.empty_cache()