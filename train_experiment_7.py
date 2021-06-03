from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int)
parser.add_argument("p", required=False, default=0, type=int)
args = parser.parse_args()

mag = 'small'
lr_max = [0.03, 0.04][args.p]
weight_decay = 3e-5
p_name = 'pod_half'

# skip_type = "res_skip"
val_fold_list = [args.gpu]
exp_list = [0]


def get_model_params(exp):
    # model_name = 'weight_decay_{:.1e}'.format(weight_decay)
    if exp == 0:
        model_name = 'res_encoder_lr_max_{}'.format(lr_max)
    elif exp == 1:
        model_name = 'res_encoder_gv_aug_mag_{}'.format(mag)
    elif exp == 2:
        model_name = 'res_encoder_weight_decay_{}'.format(weight_decay)
        
    patch_size = [32, 128, 128]
    prg_trn_sizes = [[20, 160, 160],
                     [24, 192, 192],
                     [28, 224, 224],
                     [32, 256, 256]]
    out_shape = [[20, 80, 80],
                 [24, 96, 96],
                 [28, 112, 112],
                 [32, 128, 128]]
    block = 'res'
    model_params = get_model_params_3d_nnUNet(patch_size, 2,
                                              use_prg_trn=True)
    model_params['training']['prg_trn_sizes'] = prg_trn_sizes

    del model_params['network']['kernel_sizes']
    del model_params['network']['kernel_sizes_up']
    del model_params['network']['n_pyramid_scales']
    model_params['architecture'] = 'unetresencoder'
    model_params['network']['block'] = block
    model_params['network']['z_to_xy_ratio'] = 4
    model_params['network']['stochdepth_rate'] = 0
    # this time we change the amount of augmentation during training
    prg_trn_aug_params = {}
    c = 4

    # reduce magnitude of gray value augmentations
    if mag == 'normal':
        prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.1/c], [0, 0.1]])
        prg_trn_aug_params['mm_sigma_blur'] = np.array([[0.5/c, 0.5 + 1/c], [0.5, 1.5]])
        prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.3/c, 1 + 0.3/c], [0.7, 1.3]])
        prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.35/c, 1 + 0.5/c], [0.65, 1.5]])
        prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 1/c], [1, 2]])
        prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.3/c, 1 + 0.5/c], [0.7, 1.5]])
    elif mag == 'small':
        prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.05/c], [0, 0.05]])
        prg_trn_aug_params['mm_sigma_blur'] = np.array([[0.5/c, 0.5 + 0.5/c], [0.5, 1.0]])
        prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.15/c, 1 + 0.15/c], [0.85, 1.15]])
        prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.175/c, 1 + 0.25/c], [0.825, 1.25]])
        prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 0.5/c], [1, 1.5]])
        prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.15/c, 1 + 0.25/c], [0.85, 1.25]])
    
    prg_trn_aug_params['out_shape'] = out_shape
    model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
    model_params['training']['prg_trn_resize_on_the_fly'] = False
    model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
    model_params['training']['lr_params'] = {'n_warmup_epochs': 50, 'lr_max': lr_max}
    model_params['training']['opt_params'] = {'momentum': 0.99,
                                              'weight_decay': weight_decay,
                                              'nesterov': True,
                                              'lr': 10**-2}
    return model_params, model_name


for val_fold, exp in zip(val_fold_list, exp_list):
    model_params, model_name = get_model_params(exp)
    model = SegmentationModel(val_fold=val_fold,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name=model_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set()
    model.eval_raw_data_npz('BARTS')
    model.clean()
    
    # if val_fold == 3:
    #     ens = SegmentationEnsemble(val_fold=list(range(5)),
    #                                data_name='OV04',
    #                                preprocessed_name=p_name,
    #                                model_name=model_name)
    #     if ens.all_folds_complete():
    #         ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)
