from ovseg.model.RegionexpertModel import RegionexpertModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("w", type=int)
args = parser.parse_args()


w_list = [2/3, 2/5, 2/9, 2/17, 2/33] 
w = w_list[args.w]

p_name =  'lymph_reg_expert_'+str(w)


model_name = 'U-Res-Net'

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=[8, 48, 48],
                                                     z_to_xy_ratio=5.0/0.67,
                                                     use_prg_trn=False,
                                                     larger_res_encoder=False,
                                                     n_fg_classes=1)
model_params['network']['filters'] = 32
model_params['network']['n_blocks_list'] = [4, 2, 1]
model_params['architecture'] = 'UResNet'
model_params['training']['lr_params']['lr_max'] = 0.01
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                          'dice_loss']}

model_params['data']['folders'] = ['images', 'labels', 'regions']
model_params['data']['keys'] = ['image', 'label', 'region']
model_params['training']['batches_have_masks'] = True
model_params['training']['num_epochs'] = 500
model_params['postprocessing'] = {'mask_with_reg': True}

model = RegionexpertModel(val_fold=0,
                          data_name='OV04',
                          preprocessed_name=p_name, 
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
# model.eval_validation_set()
# model.eval_training_set(force_evaluation=True)
model.eval_raw_dataset('BARTS')
