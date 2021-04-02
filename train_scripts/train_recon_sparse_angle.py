from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--angles', required=False, default='full')
parser.add_argument('--im_filters', required=False, default=5)
parser.add_argument('--data_filters', required=False, default=1)
args = parser.parse_args()

imf = 'images'
projf = 'projections_' + args.angles


model_parameters = get_model_params_2d_reconstruction('reconstruction_network_fbp_convs',
                                                      imf,
                                                      projf)
model_parameters['preprocessing']['window'] = [-31.87421739, 318]
model_parameters['preprocessing']['scaling'] = [64.64038, -1.7883005]
model_parameters['network'] = {'filters_data_space': int(args.data_filters),
                               'filters_images_space': int(args.im_filters)}

for num_epochs in [500, 1000, 1500, 2000, 2500]:
    model_parameters['training']['num_epochs'] = num_epochs
    model_name = 'recon_fbp_convs_'+str(num_epochs) + '_' + args.angles + \
        '_' + str(args.data_filters) + '_' + str(args.im_filters)
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
                                 'recon_fbp_convs_'+str(num_epochs-500) + '_' + args.angles +
                                 '_' + str(args.data_filters) + '_' + str(args.im_filters),
                                 'fold_0')
        path_to_log = os.path.join(os.environ['OV_DATA_BASE'],
                                   'trained_models',
                                   'OV04',
                                   model_name,
                                   'fold_0',
                                   'training_log.txt')
        if not os.path.exists(path_to_log):
            model.training.load_last_checkpoint(load_path)
            model.training.save_checkpoint()
    model.training.train()
    model.eval_validation_set(save_preds=True, force_evaluation=False)
    model.eval_training_set(save_preds=True, force_evaluation=False)
    torch.cuda.empty_cache()
