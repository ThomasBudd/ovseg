from ovseg.model.RestaurationModel import RestaurationModel
from ovseg.model.model_parameters_restauration import get_model_params_2d_restauration
import nibabel as nib
import numpy as np
from os import environ, mkdir, listdir
from os.path import join, exists
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

fbp_folder = ['fbps_full', 'fbps_half', 'fbps_quater', 'fbps_eights', 'fbps_16', 'fbps_32'][args.exp]


model_params = get_model_params_2d_restauration()
model_params['training']['num_epochs'] = 2500
model_params['training']['compute_val_psnr_everk_k_epochs'] = 100
model_params['data']['folders'] = ['images_restauration', fbp_folder]
model = RestaurationModel(val_fold=0,
                          data_name='OV04',
                          model_name='restauration_'+fbp_folder,
                          model_parameters=model_params,
                          preprocessed_name='pod_2d')

model.training.train()
model.eval_validation_set(save_preds=True)
model.eval_training_set(save_preds=True)

# %% now convert the predictions to preprocessed images

prep_folder = join(model.preprocessed_path, 'restaurations_'+fbp_folder[5:])
if not exists(prep_folder):
    mkdir(prep_folder)

for folder_name in ['training_fold_0', 'cross_validation']:
    pred_folder = join(environ['OV_DATA_BASE'], 'predictions', model.data_name,
                       model.model_name, folder_name)
    for case in listdir(pred_folder):
        im = nib.load(join(pred_folder, case)).get_fdata()
        im = (im - model.scaling[1]) / model.scaling[0]
        np.save(join(prep_folder, case.split('.')[0]), im)
