from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("w", type=int)
args = parser.parse_args()

w_list = 4*[0.01] + 2*[1e-3, 1e-4, 1e-5]
# w_list = 3*[1e-3, 1e-4, 1e-5]
w = w_list[args.w]
vf_list = list(range(1,5)) + 3 * [0, 1]
# vf_list = 3 * [2, 3, 4]
vf = vf_list[args.w]

model_params = get_model_params_3d_res_encoder_U_Net([32, 256, 256], 8, n_fg_classes=11,
                                                     larger_res_encoder=True)

model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg'],
                                          'loss_kwargs': [{'weight_bg': w,
                                                           'n_fg_classes': 11}]}
# model_params['data']['trn_dl_params']['mask_dist'] = [2, 15, 15]
# model_params['data']['val_dl_params']['mask_dist'] = [2, 15, 15]
model_params['training']['batches_have_masks'] = True
model_params['data']['folders'] = ['images', 'labels', 'masks']
model_params['data']['keys'] = ['image', 'label', 'mask']
model_params['data']['trn_dl_params']['mask_key'] = 'mask'
model_params['data']['val_dl_params']['mask_key'] = 'mask'

# model_params['data']['trn_dl_params']['epoch_len'] = 25
# model_params['data']['val_dl_params']['epoch_len'] = 2
# model_params['network']['filters'] = 8
# model_params['training']['num_epochs'] = 100

model = RegionfindingModel(val_fold=vf, data_name='OV04', preprocessed_name='multiclass',
                           model_name='ROIfinding_'+str(w), model_parameters=model_params)

model.training.train()
model.eval_validation_set(save_preds=True, save_plots=True)
model.eval_raw_dataset('BARTS', save_preds=True, save_plots=True)
