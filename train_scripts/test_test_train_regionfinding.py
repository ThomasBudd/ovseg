from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.model.RegionfindingEnsemble import RegionfindingEnsemble

w = 0.01
vf = 0

model_params = get_model_params_3d_res_encoder_U_Net([16, 128, 128], 8, n_fg_classes=11,
                                                     larger_res_encoder=True)

model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                          'sens_loss'],
                                          'loss_kwargs': [{'weight_bg': w,
                                                           'n_fg_classes': 11},
                                                          {'eps': 1e-5}]}
# model_params['data']['trn_dl_params']['mask_dist'] = [2, 15, 15]
# model_params['data']['val_dl_params']['mask_dist'] = [2, 15, 15]
model_params['training']['batches_have_masks'] = True
model_params['data']['folders'] = ['images', 'labels', 'masks']
model_params['data']['keys'] = ['image', 'label', 'mask']
model_params['data']['trn_dl_params']['mask_key'] = 'mask'
model_params['data']['val_dl_params']['mask_key'] = 'mask'

model_params['data']['trn_dl_params']['epoch_len'] = 250
model_params['data']['val_dl_params']['epoch_len'] = 20
model_params['network']['filters'] = 12
model_params['training']['num_epochs'] = 250

model = RegionfindingModel(val_fold=vf, data_name='OV04', preprocessed_name='multiclass',
                           model_name='delte_me', model_parameters=model_params)

model.training.train()
model.eval_validation_set(save_preds=False, save_plots=True)

