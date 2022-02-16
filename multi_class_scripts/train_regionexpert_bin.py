from ovseg.model.RegionexpertModel import RegionexpertModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.preprocessing.RegionexpertPreprocessing import RegionexpertPreprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

w_list = [0.001, 0.01, 0.1, 0.5]
model_names_ps = ['regfinding_{}'.format(w) for w in w_list] + ['regfinding_U-Net5_{}'.format(w) for w in w_list]

model_name_ps = model_names_ps[args.exp]

p_name = 'bin_reg_expert_'+model_name_ps


prep = RegionexpertPreprocessing(apply_resizing=True,
                                 apply_pooling=False,
                                 apply_windowing=True,
                                 region_finding_model={'data_name': 'OV04',
                                                       'preprocessed_name': 'bin_reg',
                                                       'model_name': model_name_ps},
                                  reduce_lb_to_single_class=True)

prep.plan_preprocessing_raw_data('OV04')
prep.preprocess_raw_data('OV04', p_name)

patch_size = [40, 320, 320]
model_name = 'U-Net5'
use_prg_trn = True
out_shape = [[24, 192, 192],
             [28, 224, 224],
             [36, 288, 288],
             [40, 320, 320]]
larger_res_encoder = True
model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/0.67,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=1,
                                                     out_shape=out_shape)

model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                          'dice_loss_weighted'],
                                          'loss_kwargs': [{'weight_bg': 1,
                                                           'n_fg_classes': 1},
                                                          {'eps': 1e-5,
                                                           'weight': 1}]}

model_params['data']['folders'] = ['images', 'labels', 'regions']
model_params['data']['keys'] = ['image', 'label', 'region']
model_params['training']['batches_have_masks'] = True
model_params['postprocessing'] = {'mask_with_reg': True}

model = RegionexpertModel(val_fold=0,
                          data_name='OV04',
                          preprocessed_name=p_name, 
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set(force_evaluation=True)
model.eval_training_set(force_evaluation=True)
model.eval_raw_dataset('BARTS', force_evaluation=True)
