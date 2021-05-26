from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import numpy as np

val_fold_list = list(range(5, 8))
p_name = 'om_half'

patch_size = [32, 128, 128]
prg_trn_sizes = [[16, 128, 128],
                 [24, 192, 192],
                 [32, 256, 256]]
out_shape = [[16, 64, 64],
             [24, 96, 96],
             [32, 128, 128]]

model_params = get_model_params_3d_nnUNet(patch_size, 2,
                                          use_prg_trn=True)
# model_params['training']['prg_trn_sizes'] = prg_trn_sizes
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
prg_trn_aug_params['out_shape'] = out_shape
# params = model_params['augmentation']['torch_params']['grid_inplane']
# for key in params:
#     if key.startswith('p'):
#         prg_trn_aug_params[key] = [params[key]/2, params[key]]
# model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
# model_params['training']['prg_trn_resize_on_the_fly'] = False
model_name = 'no_prg_trn_32_128'

for val_fold in val_fold_list:
    model = SegmentationModel(val_fold=val_fold,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name=model_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set()
    model.clean()

ens = SegmentationEnsemble(val_fold=val_fold_list,
                           data_name='OV04',
                           preprocessed_name=p_name,
                           model_name=model_name)
if ens.all_folds_complete():
    ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)