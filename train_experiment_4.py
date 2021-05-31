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
if args.gpu == 4:
    p_name = 'pod_no_resizing_half'
elif args.gpu == 5:
    p_name = 'pod_z_norm_half'


weight_decay=3e-5

# skip_type = "res_skip"
val_fold_list = list(range(5))
exp_list = 5 * [args.gpu]


def get_model_params(exp):
    # model_name = 'weight_decay_{:.1e}'.format(weight_decay)
    model_name = ['stuff_weight_decay_3e-5', 'stuff_shearing_scaling', 'stuff_no_flipping',
                  'stuff_dropout_logits', 'stuff_no_resizing', 'stuff_z_norm'][exp]
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
    model_params['augmentation']['torch_params']['grid_inplane']['p_rot'] = 0.5
    model_params['augmentation']['torch_params']['grid_inplane']['p_zoom'] = 0.5
    model_params['augmentation']['torch_params']['grid_inplane']['mm_rot'] = [-20, 20]
    model_params['augmentation']['torch_params']['grid_inplane']['mm_zoom'] = [0.8, 1.2]
    if exp == 1:
        model_params['augmentation']['torch_params']['grid_inplane']['p_scale_if_zoom'] = 0.5
        model_params['augmentation']['torch_params']['grid_inplane']['p_shear'] = 0.5
    if exp == 2:
        model_params['augmentation']['torch_params']['grid_inplane']['apply_flipping'] = False
        model_params['prediction']['mode'] = 'simple'
    if exp == 3:
        model_name['network']['p_dropout_logits'] = 0.25
        prg_trn_arch_params={'p_dropout_logits': np.array([0.25/4, 0.25])}
        model_params['training']['prg_trn_arch_params'] = prg_trn_arch_params
        

    # reduce magnitude of gray value augmentations
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
