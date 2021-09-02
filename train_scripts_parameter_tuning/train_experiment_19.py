from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int)
# parser.add_argument("p", type=int)
args = parser.parse_args()

exp = args.gpu
p_name = ['om_08_nt', 'lesions_upper', 'lesions_center', 
          'lesions_lymphnodes', 'pod_full_nt', 'multiclass'][exp]

if exp < 4:
    use_prg_trn = False
    model_name = 'res_encoder_no_prg_lrn' 
    patch_size = [32, 216, 216]
    out_shape = [[20, 136, 136],
                 [24, 168, 168],
                 [28, 192, 192],
                 [32, 216, 216]]
    
    prg_trn_sizes = np.array(out_shape)
    prg_trn_sizes[:, 1:] *= 2
    model_params = get_model_params_3d_nnUNet(patch_size, 2,
                                              use_prg_trn=use_prg_trn)

    del model_params['network']['kernel_sizes']
    del model_params['network']['kernel_sizes_up']
    del model_params['network']['n_pyramid_scales']
    model_params['architecture'] = 'unetresencoder'
    model_params['network']['block'] = 'res'
    model_params['network']['z_to_xy_ratio'] = 6.25
    model_params['network']['stochdepth_rate'] = 0
    
    # this time we change the amount of augmentation during training
    prg_trn_aug_params = {}
    c = 4

    prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.1/c], [0, 0.1]])
    prg_trn_aug_params['mm_sigma_blur'] = np.array([[0.5/c, 0.5 + 1/c], [0.5, 1.5]])
    prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.3/c, 1 + 0.3/c], [0.7, 1.3]])
    prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.35/c, 1 + 0.5/c], [0.65, 1.5]])
    prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 1/c], [1, 2]])
    prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.3/c, 1 + 0.5/c], [0.7, 1.5]])
    prg_trn_aug_params['out_shape'] = out_shape
    if use_prg_trn:
        model_params['training']['prg_trn_sizes'] = prg_trn_sizes
        model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
        model_params['training']['prg_trn_resize_on_the_fly'] = False
    model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
    model_params['training']['lr_params'] = {'n_warmup_epochs': 50, 'lr_max': 0.02}
    model_params['training']['opt_params'] = {'momentum': 0.99,
                                              'weight_decay': 3e-5,
                                              'nesterov': True,
                                              'lr': 2*10**-2}
elif exp == 4:
    use_prg_trn = True
    model_name = 'larger_res_encoder'

    patch_size = [32, 256, 256]
    out_shape = [[20, 160, 160],
                 [24, 192, 192],
                 [28, 224, 224],
                 [32, 256, 256]]
        

    prg_trn_sizes = np.array(out_shape)
    prg_trn_sizes[:, 1:] *= 2
    model_params = get_model_params_3d_nnUNet(patch_size, 2,
                                              use_prg_trn=use_prg_trn)

    del model_params['network']['kernel_sizes']
    del model_params['network']['kernel_sizes_up']
    del model_params['network']['n_pyramid_scales']
    model_params['architecture'] = 'unetresencoder'
    model_params['network']['block'] = 'res'
    model_params['network']['z_to_xy_ratio'] = 8
    model_params['network']['stochdepth_rate'] = 0
    model_params['network']['filters'] = 16
    model_params['network']['n_blocks_list'] = [1, 1, 2, 6, 3]
    
    # this time we change the amount of augmentation during training
    prg_trn_aug_params = {}
    c = 4

    prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.1/c], [0, 0.1]])
    prg_trn_aug_params['mm_sigma_blur'] = np.array([[0.5/c, 0.5 + 1/c], [0.5, 1.5]])
    prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.3/c, 1 + 0.3/c], [0.7, 1.3]])
    prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.35/c, 1 + 0.5/c], [0.65, 1.5]])
    prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 1/c], [1, 2]])
    prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.3/c, 1 + 0.5/c], [0.7, 1.5]])
    prg_trn_aug_params['out_shape'] = out_shape
    if use_prg_trn:
        model_params['training']['prg_trn_sizes'] = prg_trn_sizes
        model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
        model_params['training']['prg_trn_resize_on_the_fly'] = False
    model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
    model_params['training']['lr_params'] = {'n_warmup_epochs': 50, 'lr_max': 0.02}
    model_params['training']['opt_params'] = {'momentum': 0.99,
                                              'weight_decay': 3e-5,
                                              'nesterov': True,
                                              'lr': 2*10**-2}
else:
    use_prg_trn = False
    model_name = 'res_encoder_no_prg_lrn' 
    patch_size = [32, 216, 216]
    out_shape = [[20, 136, 136],
                 [24, 168, 168],
                 [28, 192, 192],
                 [32, 216, 216]]
    
    prg_trn_sizes = np.array(out_shape)
    prg_trn_sizes[:, 1:] *= 2
    model_params = get_model_params_3d_nnUNet(patch_size, 2,
                                              use_prg_trn=use_prg_trn,
                                              n_fg_classes=11)

    del model_params['network']['kernel_sizes']
    del model_params['network']['kernel_sizes_up']
    del model_params['network']['n_pyramid_scales']
    model_params['architecture'] = 'unetresencoder'
    model_params['network']['block'] = 'res'
    model_params['network']['z_to_xy_ratio'] = 6.25
    model_params['network']['stochdepth_rate'] = 0
    
    # this time we change the amount of augmentation during training
    prg_trn_aug_params = {}
    c = 4

    prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.1/c], [0, 0.1]])
    prg_trn_aug_params['mm_sigma_blur'] = np.array([[0.5/c, 0.5 + 1/c], [0.5, 1.5]])
    prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.3/c, 1 + 0.3/c], [0.7, 1.3]])
    prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.35/c, 1 + 0.5/c], [0.65, 1.5]])
    prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 1/c], [1, 2]])
    prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.3/c, 1 + 0.5/c], [0.7, 1.5]])
    prg_trn_aug_params['out_shape'] = out_shape
    if use_prg_trn:
        model_params['training']['prg_trn_sizes'] = prg_trn_sizes
        model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
        model_params['training']['prg_trn_resize_on_the_fly'] = False
    model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
    model_params['training']['lr_params'] = {'n_warmup_epochs': 50, 'lr_max': 0.02}
    model_params['training']['opt_params'] = {'momentum': 0.99,
                                              'weight_decay': 3e-5,
                                              'nesterov': True,
                                              'lr': 2*10**-2}


for val_fold in range(5):
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

ens.eval_raw_dataset('BARTS', save_preds=True)
