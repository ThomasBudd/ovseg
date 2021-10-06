from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.model.RegionfindingEnsemble import RegionfindingEnsemble

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("fold", type=int)
parser.add_argument("exp", type=int)
args = parser.parse_args()

w = 0.1
vf = args.fold


p_name = ['mesentery_reg', 'small_reg'][args.exp]
patch_size = [40, 320, 320]
model_params = get_model_params_3d_res_encoder_U_Net(patch_size, 
                                                     5/0.67,
                                                     n_fg_classes=1,
                                                     larger_res_encoder=True)

model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                          'dice_loss_weighted'],
                                          'loss_kwargs': [{'weight_bg': w,
                                                           'n_fg_classes': 11},
                                                          {'eps': 1e-5,
                                                           'weight': w}]}
model_params['data']['folders'] = ['images', 'labels', 'regions']
model_params['data']['keys'] = ['image', 'label', 'region']
# we train using the regions as ground truht we're training for
model_params['data']['trn_dl_params']['label_key'] = 'region'
model_params['data']['val_dl_params']['label_key'] = 'region'
model_params['training']['num_epochs'] = 500

model = RegionfindingModel(val_fold=vf,
                           data_name='OV04',
                           preprocessed_name=p_name,
                           model_name='regfinding_'+str(w),
                           model_parameters=model_params)

model.training.train()
model.eval_validation_set(save_preds=True, save_plots=False)
# model.eval_raw_dataset('BARTS', save_preds=True)
model.eval_raw_data_npz('BARTS')
