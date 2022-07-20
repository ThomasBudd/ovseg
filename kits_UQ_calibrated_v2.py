from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.SegmentationEnsembleV2 import SegmentationEnsembleV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
import argparse
import numpy as np
import os
from ovseg import OV_PREPROCESSED

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()
 
data_name = 'kits21_trn'
preprocessed_name = 'disease_3_1'
bs = 4
wd = 1e-4
lr_max = 0.02
patch_size = [32, 112, 112]

sizes = 16*np.round(patch_size[2] / np.arange(4,0,-1)**(1/3) / 16)
sizesz = 4*np.round(patch_size[0] / np.arange(4,0,-1)**(1/3) / 4)
out_shape = [ [int(sz),  int(s), int(s)] for s, sz in zip(sizes, sizesz)]
print(out_shape)

w1 = -1
w2 = -2

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=3.0/0.8,
                                                     use_prg_trn=True,
                                                     out_shape=out_shape,
                                                     larger_res_encoder=False,
                                                     n_fg_classes=2)

model_params['architecture'] = 'UNet'
model_params['network']['kernel_sizes'] = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
model_params['network']['in_channels'] = 1
model_params['network']['norm'] = 'inst'
del model_params['network']['block']
del model_params['network']['z_to_xy_ratio']
del model_params['network']['n_blocks_list']
del model_params['network']['stochdepth_rate']

model_params['data']['folders'] = ['images', 'labels', 'masks']
model_params['data']['keys'] = ['image', 'label', 'mask']
model_params['data']['n_folds'] = 3

for s in ['trn_dl_params', 'val_dl_params']:
    model_params['data'][s]['batch_size'] = bs
    model_params['data'][s]['min_biased_samples'] = 1
    # model_params['data'][s]['num_workers'] = 14
    del model_params['data'][s]['store_coords_in_ram']
    del model_params['data'][s]['memmap']
model_params['training']['batches_have_masks'] = True
model_params['training']['opt_params']['weight_decay'] = wd
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['stop_after_epochs'] = [750]
model_params['training']['lr_params']['lr_max'] = lr_max
model_params['training']['loss_params']['loss_names'] = ['dice_loss_sigm_weighted',
                                                         'cross_entropy_exp_weight']
model_params['training']['loss_params']['loss_kwargs'] = 2*[{'w_list':[0, 0]}]
model_params['postprocessing'] = {'mask_with_reg': True}

w = 0

# %%
    
model_name = f'UQ_calibrated_{w:.2f}'
model_params['training']['loss_params']['loss_kwargs'] = 2*[{'w_list':[w+w1, w+w2]}]
model_params['training']['stop_after_epochs'] = []

model = SegmentationModelV2(val_fold=args.vf,
                            data_name=data_name,
                            model_name=model_name,
                            preprocessed_name=preprocessed_name,
                            model_parameters=model_params)
path_to_checkpoint = os.path.join(os.environ['OV_DATA_BASE'],
                                  'trained_models',
                                  data_name,
                                  preprocessed_name,
                                  model_name,
                                  f'fold_{args.vf}',
                                  'attribute_checkpoint.pkl')
if os.path.exists(path_to_checkpoint):
    print('Previous checkpoint found and loaded')
else:
    print('Loading pretrained checkpoint')
    path_to_checkpoint = os.path.join(os.environ['OV_DATA_BASE'],
                                        'trained_models',
                                        data_name,
                                        preprocessed_name,
                                        'stopped',
                                        f'fold_{args.vf}')
    model.training.load_last_checkpoint(path_to_checkpoint)
    model.training.loss_params = {'loss_names': ['dice_loss_sigm_weighted',
                                                 'cross_entropy_exp_weight'],
                                  'loss_kwargs': 2*[{'w_list':[w,w]}]}
    model.training.initialise_loss()
    model.training.save_checkpoint()
    model.training.prg_trn_update_parameters()
model.training.train()
model.eval_raw_dataset('kits21_tst')
