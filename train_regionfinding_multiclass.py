from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.model.RegionfindingEnsemble import RegionfindingEnsemble
from ovseg.preprocessing.RegionexpertPreprocessing import RegionexpertPreprocessing
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

w = 0
vf = args.vf

p_name = 'multiclass_reg'
patch_size = [40, 320, 320]
use_prg_trn = True
out_shape = [[24, 192, 192],
             [28, 224, 224],
             [36, 288, 288],
             [40, 320, 320]]
larger_res_encoder = True
model_params = get_model_params_3d_res_encoder_U_Net(patch_size, 
                                                     5/0.67,
                                                     n_fg_classes=6,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     out_shape=out_shape)

model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                          'dice_loss_weighted'],
                                          'loss_kwargs': [{'weight_bg': 0,
                                                           'n_fg_classes': 6},
                                                          {'eps': 1e-5,
                                                           'weight': 0}]}
model_params['data']['folders'] = ['images', 'labels', 'regions']
model_params['data']['keys'] = ['image', 'label', 'region']
# we train using the regions as ground truht we're training for
for dl_str in ['trn_dl_params', 'val_dl_params']:
    model_params['data'][dl_str]['label_key'] = 'region'
    model_params['data'][dl_str]['bias'] = 'cl_fg'
    model_params['data'][dl_str]['n_fg_classes'] = 6
    model_params['data'][dl_str]['min_biased_samples'] = 2
model_params['network']['use_logit_bias'] = True

model = RegionfindingModel(val_fold=vf,
                           data_name='OV04',
                           preprocessed_name=p_name,
                           model_name='regfinding_'+str(w),
                           model_parameters=model_params)

# manually set the logits to 0 and biases to -50 for all background outputs
for log_layer in model.network.all_logits:
    w = log_layer.logits.weight.clone()
    w[0] = 0
    log_layer.logits.weight = torch.nn.Parameter(w)
    b = log_layer.logits.bias.clone()
    b[0] = -50
    log_layer.logits.bias = torch.nn.Parameter(b)

model.training.train()
model.eval_validation_set(save_preds=True, save_plots=False)
model.eval_raw_data_npz('BARTS')

# w = w_list[vf]
ens = RegionfindingEnsemble(val_fold=list(range(5)), 
                            data_name='OV04',
                            preprocessed_name=p_name,
                            model_name='regfinding_'+str(w))
if ens.all_folds_complete():
    ens.eval_raw_dataset('BARTS', save_preds=True)

    prep = RegionexpertPreprocessing(apply_resizing=True,
                                     apply_pooling=False,
                                     apply_windowing=True,
                                     region_finding_model={'data_name': 'OV04',
                                                           'preprocessed_name': p_name,
                                                           'model_name': 'regfinding_'+str(w)},
                                     lb_classes=[13, 15, 17],
                                     target_spacing=[5.0, 0.67, 0.67],
                                     save_only_fg_scans=False)
    
    prep.plan_preprocessing_raw_data('OV04')
    prep.preprocess_raw_data('OV04', 'lymph_reg_expert_'+str(w))

    prep = RegionexpertPreprocessing(apply_resizing=True,
                                     apply_pooling=False,
                                     apply_windowing=True,
                                     region_finding_model={'data_name': 'OV04',
                                                           'preprocessed_name': p_name,
                                                           'model_name': 'regfinding_'+str(w)},
                                     lb_classes=[1],
                                     target_spacing=[5.0, 0.67, 0.67],
                                     save_only_fg_scans=False)
    
    prep.plan_preprocessing_raw_data('OV04')
    prep.preprocess_raw_data('OV04', 'om_reg_expert_'+str(w))

