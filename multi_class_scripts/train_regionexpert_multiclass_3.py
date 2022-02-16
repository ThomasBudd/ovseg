from ovseg.model.RegionexpertModel import RegionexpertModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.model.RegionexpertEnsemble import RegionexpertEnsemble
from time import sleep

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

w = 0.001

patch_size = [40, 320, 320]
use_prg_trn = True
out_shape = [[24, 192, 192],
             [28, 224, 224],
             [36, 288, 288],
             [40, 320, 320]]
larger_res_encoder = True

# %%
lb_classes = [1, 9]
vf = args.vf
model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/0.67,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=1,
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                          'dice_loss']}
model_params['data']['folders'] = ['images', 'labels', 'regions']
model_params['data']['keys'] = ['image', 'label', 'region']
model_params['training']['batches_have_masks'] = True
model_params['postprocessing'] = {'mask_with_reg': True}
model_params['data']['val_dl_params']['bias'] = 'cl_fg'
model_params['data']['trn_dl_params']['bias'] = 'cl_fg'
model_params['data']['val_dl_params']['n_fg_classes'] = 2
model_params['data']['trn_dl_params']['n_fg_classes'] = 2

model_name = 'U-Net5'
p_name = 'reg_expert_'+'_'.join([str(c) for c in lb_classes])
model = RegionexpertModel(val_fold=vf,
                          data_name='OV04',
                          preprocessed_name=p_name, 
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_raw_data_npz('BARTS')

ens = RegionexpertEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name=p_name, 
                           model_name='U-Net5')

ens.wait_until_all_folds_complete()
ens.eval_raw_dataset('BARTS')