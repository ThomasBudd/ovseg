from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
import torch
import os
num_epochs = 500

val_fold = 0
data_name = 'OV04'

if not os.path.exists(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', data_name,
                                   'pod_default', 'projections')):
    from ovseg.preprocessing.Reconstruction2dSimPreprocessing import Reconstruction2dSimPreprocessing
    from ovseg.networks.recon_networks import get_operator

    operator = get_operator()
    preprocessing = Reconstruction2dSimPreprocessing(operator, window=[-50, 350])
    preprocessing.preprocess_raw_folders(['OV04', 'BARTS', 'ApolloTCGA'],
                                         'pod_default',
                                         data_name=data_name,
                                         proj_folder_name='projections',
                                         im_folder_name='images_att')

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
