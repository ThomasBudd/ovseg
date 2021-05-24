from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
import numpy as np

model_params = get_model_params_3d_nnUNet([32, 128, 128], 2,
                                          use_prg_trn=True)
model_params['network']['filters'] = 8
model_params['training']['num_epochs'] = 100
model_params['data']['trn_dl_params']['epoch_len'] = 25
prg_trn_sizes = [[16, 128, 128],
                 [24, 192, 192],
                 [32, 256, 256]]
out_shape = [[16, 64, 64],
             [24, 96, 96],
             [32, 128, 128]]
model_params['training']['prg_trn_sizes'] = prg_trn_sizes
prg_trn_aug_params = {}
c = 4
prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.1/c], [0, 0.1]])
prg_trn_aug_params['mm_sigma_blur'] = np.array([[1 - 0.5/c, 1 + 0.5/c], [0.5, 1.5]])
prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.3/c, 1 + 0.3/c], [0.7, 1.3]])
prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.45/c, 1 + 0.5/c], [0.65, 1.5]])
prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 1/c], [1, 2]])
prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.3/c, 1 + 0.5/c], [0.7, 1.5]])
prg_trn_aug_params['out_shape'] = out_shape
# params = model_params['augmentation']['torch_params']['grid_inplane']
# for key in params:
#     if key.startswith('p'):
#         prg_trn_aug_params[key] = [params[key]/2, params[key]]
model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
model_params['training']['prg_trn_resize_on_the_fly'] = True
model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
model_params['training']['lr_params'] = None
model_params['training']['no_bias_weight_decay'] = True
model_name = 'debug_no_bias_weight_decay_v2'
val_fold = 0
p_name = 'pod_half'
model = SegmentationModel(val_fold=val_fold,
                          data_name='OV04_test',
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()