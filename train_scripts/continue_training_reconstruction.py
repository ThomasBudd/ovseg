from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data')
args = parser.parse_args()
data_name = args.data

if data_name == 'OV04':
    preprocessed_name = 'pod_default'
else:
    preprocessed_name = 'default'

projf = 'projections_normal'
imf = 'images_HU_rescale'
model_params = get_model_params_2d_reconstruction('reconstruction_network_fbp_convs',
                                                  imf,
                                                  projf)
model_params['training']['num_epochs'] = 1000
model_params['training']['opt_params']['lr'] = 0.5 * 10 ** -4
model_params['data']['trn_dl_params']['n_bias'] = 3

model = Reconstruction2dSimModel(val_fold=0,
                                 data_name=data_name,
                                 model_name='recon_fbp_convs_pretrained_continued',
                                 model_parameters=model_params)

path = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', data_name,
                    'recon_fbp_convs_pretrained', 'fold_0')
model.training.load_last_checkpoint(path)
model.training.train()
model.eval_validation_set(save_preds=True, force_evaluation=True)
model.eval_training_set(save_preds=True, force_evaluation=True)