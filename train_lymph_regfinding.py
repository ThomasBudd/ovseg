from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.model.RegionfindingEnsemble import RegionfindingEnsemble
from time import sleep

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

w_list = [2/3, 2/5, 2/9, 2/17, 2/33] 
vf = args.vf


p_name = 'lymph_reg'
patch_size = [40, 320, 320]
model_params = get_model_params_3d_res_encoder_U_Net(patch_size, 
                                                     5/0.67,
                                                     n_fg_classes=3,
                                                     larger_res_encoder=True)

for w in w_list:
    model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                              'dice_loss_weighted'],
                                              'loss_kwargs': [{'weight_bg': w,
                                                               'n_fg_classes': 3},
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

w = w_list[vf]
ens = RegionfindingEnsemble(val_fold=list(range(5)), 
                            data_name='OV04',
                            preprocessed_name=p_name,
                            model_name='regfinding_'+str(w))
while not ens.all_folds_complete():
    sleep(60)
ens.eval_raw_dataset('BARTS')
