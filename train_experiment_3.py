from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int)
# parser.add_argument("rep", required=False, default=0, type=int)
args = parser.parse_args()
p_name = 'pod_half'

# skip_type = "res_skip"
val_fold_list = list(range(5))
exp_list = 5 * [args.gpu]


def get_model_params(exp):
    p = [0.2, 0.2, 0.5, 1, 0.5, 1][exp]
    mag = ['normal', 'normal', 'normal', 'normal', 'small', 'small'][exp]
    weight_decay = [1e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5][exp]
    # model_name = 'weight_decay_{:.1e}'.format(weight_decay)
    model_name = 'spatial_aug_{}_{}_{}'.format(p, mag, weight_decay)#, args.rep)
    patch_size = [32, 128, 128]
    prg_trn_sizes = [[16, 128, 128],
                     [24, 192, 192],
                     [32, 256, 256]]
    out_shape = [[16, 64, 64],
                 [24, 96, 96],
                 [32, 128, 128]]

    model_params = get_model_params_3d_nnUNet(patch_size, 2,
                                              use_prg_trn=True)
    model_params['training']['prg_trn_sizes'] = prg_trn_sizes
    
    # this time we change the amount of augmentation during training
    prg_trn_aug_params = {}
    c = 4
    model_params['augmentation']['torch_params']['grid_inplane']['p_rot'] = p
    model_params['augmentation']['torch_params']['grid_inplane']['p_zoom'] = p
    if mag == 'small':
        model_params['augmentation']['torch_params']['grid_inplane']['mm_rot'] = [-20, 20]
        model_params['augmentation']['torch_params']['grid_inplane']['mm_zoom'] = [0.8, 1.2]

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
    model_params['training']['lr_params'] = {'n_warmup_epochs': 50, 'lr_max': 0.02}
    model_params['training']['opt_params'] = {'momentum': 0.99,
                                              'weight_decay': weight_decay,
                                              'nesterov': True,
                                              'lr': 10**-2}
    model_params['data']['trn_dl_params']['num_workers'] = 16
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
    model.clean()

ens = SegmentationEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name=p_name,
                           model_name=model_name)
if ens.all_folds_complete():
    ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)
