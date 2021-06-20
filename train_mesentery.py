from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int)
# parser.add_argument("p", type=int)
args = parser.parse_args()

# skip_type = "res_skip"
val_fold_list = 2*[args.gpu]
exp_list = [0, 1]

def get_model_params(exp):
    # model_name = 'larger_res_encoder'
    # if exp == 1:
    #     model_name += '_less_gv_aug'
    filters = [8, 16][exp]
    model_name = 'UNet_{}'.format(filters)
    patch_size = [32, 256, 256]
    out_shape = [[20, 160, 160],
                 [24, 192, 192],
                 [28, 224, 224],
                 [32, 256, 256]]
            
    
    prg_trn_sizes = np.array(out_shape)
    prg_trn_sizes[:, 1:] *= 2
    model_params = get_model_params_3d_nnUNet(patch_size, 2,
                                              use_prg_trn=True)
    model_params['network']['filters'] = filters
    model_params['augmentation']['torch_params']['grid_inplane']['p_rot'] = 0.5
    model_params['augmentation']['torch_params']['grid_inplane']['p_scale'] = 0.5
    # del model_params['network']['kernel_sizes']
    # del model_params['network']['kernel_sizes_up']
    # del model_params['network']['n_pyramid_scales']
    # model_params['architecture'] = 'unetresencoder'
    # model_params['network']['block'] = 'res'
    # model_params['network']['z_to_xy_ratio'] = 8
    # model_params['network']['stochdepth_rate'] = 0
    # model_params['network']['filters'] = 16
    # model_params['network']['n_blocks_list'] = [1, 1, 2, 6, 3]
    
    # this time we change the amount of augmentation during training
    prg_trn_aug_params = {}
    c = 4
    # prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.1/c], [0, 0.1]])
    # prg_trn_aug_params['mm_sigma_blur'] = np.array([[0.5/c, 0.5 + 1/c], [0.5, 1.5]])
    # prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.3/c, 1 + 0.3/c], [0.7, 1.3]])
    # prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.35/c, 1 + 0.5/c], [0.65, 1.5]])
    # prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 1/c], [1, 2]])
    # prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.3/c, 1 + 0.5/c], [0.7, 1.5]])
    # else:
    prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.05/c], [0, 0.05]])
    prg_trn_aug_params['mm_sigma_blur'] = np.array([[0.25/c, 0.25 + 0.5/c], [0.25, 0.75]])
    prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.15/c, 1 + 0.15/c], [0.85, 1.15]])
    prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.2/c, 1 + 0.25/c], [0.8, 1.25]])
    prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 0.5/c], [1, 0.5]])
    prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.15/c, 1 + 0.25/c], [0.85, 1.25]])
    prg_trn_aug_params['out_shape'] = out_shape
    model_params['training']['prg_trn_sizes'] = prg_trn_sizes
    model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
    model_params['training']['prg_trn_resize_on_the_fly'] = False
    model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
    model_params['training']['lr_params'] = {'n_warmup_epochs': 50, 'lr_max': 0.02}
    model_params['training']['opt_params'] = {'momentum': 0.99,
                                              'weight_decay': 1e-4,
                                              'nesterov': True,
                                              'lr': 2*10**-2}
    return model_params, model_name

for val_fold, exp in zip(val_fold_list, exp_list):
    model_params, model_name = get_model_params(exp)
    p_name = 'default'
    model = SegmentationModel(val_fold=val_fold,
                              data_name='mesentery',
                              preprocessed_name=p_name,
                              model_name=model_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set(save_plots=True)
    model.eval_training_set()
    model.clean()
