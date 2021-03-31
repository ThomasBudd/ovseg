from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--angles', required=False, default='full')
args = parser.parse_args()

imf = 'images'
projf = 'projections_' + args.angles


model_parameters = get_model_params_2d_reconstruction('reconstruction_network_fbp_convs',
                                                      imf,
                                                      projf)
model_parameters['preprocessing']['window'] = [-31.87421739, 318]
model_parameters['preprocessing']['scaling'] = [64.64038, -1.7883005]

for num_epochs in [500, 1000, 1500, 2000, 2500]:
    model_parameters['training']['num_epochs'] = num_epochs
    model_name = 'recon_fbp_convs_'+str(num_epochs)
    model = Reconstruction2dSimModel(val_fold=0,
                                     data_name='OV04',
                                     model_name=model_name,
                                     model_parameters=model_parameters,
                                     preprocessed_name='pod_no_resizing')
    model.save_model_parameters()
    if num_epochs > 500:
        load_path = os.path.join(os.environ['OV_DATA_BASE'],
                                 'trained_models',
                                 'OV04',
                                 'recon_fbp_convs_'+str(num_epochs-500),
                                 'fold_0')
        model.training.load_last_checkpoint(load_path)
    model.training.train()
    model.eval_validation_set(save_preds=True, force_evaluation=True)
    model.eval_training_set(save_preds=True, force_evaluation=True)
    torch.cuda.empty_cache()
