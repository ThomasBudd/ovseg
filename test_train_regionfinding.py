from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("w", type=int)
args = parser.parse_args()

w = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1][args.w]

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

model = RegionfindingModel(val_fold=0, data_name='OV04', preprocessed_name='multiclass',
                           model_name='ROIfinding_'+str(w), model_parameters=model_params)

model.training.train()
model.eval_validation_set(save_preds=True, save_plots=True)
model.eval_raw_dataset('BARTS', save_preds=True, save_plots=True)
