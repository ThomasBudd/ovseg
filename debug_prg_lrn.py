from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
import numpy as np

model_name = 'debug_prg_trn'

model_params = get_model_params_3d_nnUNet([40, 160, 160], 2,
                                          use_prg_trn=True)
model_params['training']['prg_trn_sizes'] = [[40, 128, 128],
                                             [40, 192, 192],
                                             [40, 256, 256],
                                             [40, 320, 320]]
model_params['training']['num_epochs'] = 8
# this time we change the amount of augmentation during training
prg_trn_aug_params = {}
# params = model_params['augmentation']['torch_params']['grayvalue']
# for key in params:
#     if key.startswith('p'):
#         prg_trn_aug_params[key] = [params[key]/2, params[key]]
# factor we use for reducing the magnitude of the gray value augmentations
c = 4
prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.1/c], [0, 0.1]])
prg_trn_aug_params['mm_sigma_blur'] = np.array([[1 - 0.5/c, 1 + 0.5/c], [0.5, 1.5]])
prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.3/c, 1 + 0.3/c], [0.7, 1.3]])
prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.45/c, 1 + 0.5/c], [0.65, 1.5]])
prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 1/c], [1, 2]])
prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.3/c, 1 + 0.5/c], [0.7, 1.5]])
prg_trn_aug_params['out_shape'] = [[40, 64, 64],
                                   [40, 96, 96],
                                   [40, 128, 128],
                                   [40, 160, 160]]
# params = model_params['augmentation']['torch_params']['grid_inplane']
# for key in params:
#     if key.startswith('p'):
#         prg_trn_aug_params[key] = [params[key]/2, params[key]]
model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params


p_name = 'pod_half'
val_fold = 0
model = SegmentationModel(val_fold=val_fold,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
   