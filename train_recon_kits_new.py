from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
from ovseg.preprocessing.Reconstruction2dSimPreprocessing import Reconstruction2dSimPreprocessing
from ovseg.networks.recon_networks import get_operator
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dose", required=True)
parser.add_argument("-w", '--window', required=False, default=False, action="store_true")
args = parser.parse_args()

dose = args.dose

if dose == 'normal':
    n_photons = 2 * 10**6
elif dose == 'high':
    n_photons = 3 * 10**7
else:
    raise ValueError('Unkown input {} for dose'.format(dose))

window = [-100, 350] if args.window else None
win_str = '_win' if args.window else ''

model_params = get_model_params_2d_reconstruction(image_folder='images_HU'+win_str+'_rescale',
                                                  projection_folder='projections_'+dose+win_str)
model_params['preprocessing']['window'] = window
model_params['preprocessing']['num_photons'] = n_photons
data_name = 'kits19'
preprocessed_name = 'default'
val_fold = 0
if not os.path.exists(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', data_name,
                                   preprocessed_name, 'projections_'+dose+win_str)):
    op = get_operator()
    preprocessing = Reconstruction2dSimPreprocessing(op, **model_params['preprocessing'])
    preprocessing.preprocess_raw_folders(['kits19'],
                                         preprocessed_name=preprocessed_name,
                                         data_name=data_name,
                                         proj_folder_name='projections_'+dose+win_str,
                                         im_folder_name='images_HU'+win_str+'_rescale')

model = Reconstruction2dSimModel(val_fold=val_fold,
                                 data_name=data_name,
                                 preprocessed_name=preprocessed_name,
                                 model_parameters=model_params,
                                 model_name='recon_fbp_convs_'+dose+win_str,
                                 plot_window=[-100, 350])
model.training.train()

model.eval_validation_set(save_preds=True)
model.eval_training_set(save_preds=True)
