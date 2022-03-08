from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
import numpy as np

N = 2
M = 10
model_params = get_model_params_3d_nnUNet([32, 128, 128], 2,
                                          use_prg_trn=True)
del model_params['augmentation']['torch_params']['grayvalue']
model_params['augmentation']['torch_params']['myRandAugment'] = {'n': 1, 'm': 10}
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
prg_trn_aug_params['m'] = np.array([M/4, M])
prg_trn_aug_params['out_shape'] = out_shape
model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
model_params['training']['prg_trn_resize_on_the_fly'] = True
model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
model_params['training']['lr_params'] = {'n_warmup_epochs': 50, 'lr_max': 0.02}

model_params['training']['no_bias_weight_decay'] = True
model_name = 'debug_myRandAugment_{}_{}'.format(N, M)
val_fold = 0
p_name = 'pod_half'
model = SegmentationModel(val_fold=val_fold,
                          data_name='OV04_test',
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()