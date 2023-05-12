import os
import torch
import numpy as np
import pickle


def get_model_params_2d_segmentation(n_fg_classes=1,
                                     fp32=False):
    model_parameters = {}
    # we're doing no preprocessing parameters here as they can be loaded
    # from the preprocessed folder

    # Let's jump straight to the augmentation
    grayvalue_params = {'p_noise': 0.15,
                        'p_blur': 0.1,
                        'p_bright': 0.15,
                        'p_contr': 0.15,
                        'p_low_res': 0.125,
                        'p_gamma': 0.15,
                        'p_gamma_invert': 0.15,
                        'mm_var_noise': [0, 0.1],
                        'mm_sigma_blur': [0.5, 1.5],
                        'mm_bright': [0.7, 1.3],
                        'mm_contr': [0.65, 1.5],
                        'mm_low_res': [1, 2],
                        'mm_gamma': [0.7, 1.5],
                        'n_im_channels': 1
                        }
    spatial_params = {'p_rot': 0.2,
                      'p_zoom': 0.2,
                      'p_transl': 0,
                      'p_shear': 0,
                      'mm_zoom': [0.7, 1.4],
                      'mm_rot': [-180, 180],
                      'mm_transl': [-0.25, 0.25],
                      'mm_shear': [-0.2, 0.2],
                      'apply_flipping': True,
                      'n_im_channels': 1,
                      'out_shape': None}

    model_parameters['augmentation'] = {'torch_params': {'grid_inplane': spatial_params,
                                                         'grayvalue': grayvalue_params}}

    # now the network parameters. classic 2d UNet
    model_parameters['architecture'] = 'UNet'
    network_parameters = {'in_channels': 1,
                          'out_channels': n_fg_classes + 1,
                          'kernel_sizes': 7 * [3],
                          'is_2d': True,
                          'filters': 32,
                          'filters_max': 320,
                          'n_pyramid_scales': None,
                          'conv_params': None,
                          'norm': None,
                          'norm_params': None,
                          'nonlin_params': None,
                          'kernel_sizes_up': None}
    model_parameters['network'] = network_parameters

    # we have no postprocessing parameters and just go with the default
    # now data
    ds_params = {}
    num_workers = 0 if os.name == 'nt' else 5
    trn_dl_params = {'patch_size': [512, 512],
                     'batch_size': 12,
                     'num_workers': num_workers,
                     'pin_memory': torch.cuda.is_available(),
                     'epoch_len': 250,
                     'p_bias_sampling': 0,
                     'min_biased_samples': 3,
                     'padded_patch_size': None,
                     'store_coords_in_ram': True,
                     'memmap': 'r',
                     'n_im_channels': 1,
                     'store_data_in_ram': False,
                     'return_fp16': not fp32,
                     'n_max_volumes': None}
    val_dl_params = trn_dl_params.copy()
    val_dl_params['epoch_len'] = 8
    val_dl_params['store_data_in_ram'] = True
    val_dl_params['n_max_volumes'] = 16
    val_dl_params['num_workers'] = 0   
    # as here we can store the data in RAM, 0 workers are faster
    keys = ['image', 'label']
    folders = ['images', 'labels']
    data_params = {'n_folds': 5,
                   'fixed_shuffle': True,
                   'ds_params': ds_params,
                   'trn_dl_params': trn_dl_params,
                   'keys': keys,
                   'val_dl_params': val_dl_params,
                   'folders': folders}
    model_parameters['data'] = data_params

    # prediction object
    prediction_params = {'patch_size': [512, 512],
                         'batch_size': 1,
                         'overlap': 0.5,
                         'fp32': fp32,
                         'patch_weight_type': 'gaussian',
                         'sigma_gaussian_weight': 1/8,
                         'mode': 'flip'}
    model_parameters['prediction'] = prediction_params

    # # now finally the training!
    # loss_params = {'eps': 1e-5,
    #                'dice_weight': 1.0,
    #                'ce_weight': 1.0,
    #                'pyramid_weight': 0.5}
    loss_params = {'loss_names': ['cross_entropy', 'dice_loss']}
    opt_params = {'momentum': 0.99, 'weight_decay': 3e-5, 'nesterov': True,
                  'lr': 10**-2}
    lr_params = {'beta': 0.9, 'lr_min': 0}
    training_params = {'loss_params': loss_params,
                       'num_epochs': 1000, 'opt_params': opt_params,
                       'lr_params': lr_params, 'nu_ema_trn': 0.99,
                       'nu_ema_val': 0.9, 'fp32': fp32,
                       'p_plot_list': [1, 0.5, 0.2], 'opt_name': 'SGD'}
    model_parameters['training'] = training_params
    model_parameters['prediction_key'] = 'learned_segmentation'
    return model_parameters


def get_model_params_3d_UNet(patch_size,
                             n_2d_convs,
                             use_prg_trn=False,
                             n_fg_classes=1,
                             fp32=False,
                             out_shape=None):
    model_params = get_model_params_2d_segmentation(n_fg_classes=n_fg_classes,
                                                    fp32=fp32)

    # first determine the number of stages plus the kernel sizes used there
    i = 1
    kernel_sizes = [(1, 3, 3) if 0 < n_2d_convs else (3, 3, 3)]
    psb = np.array(patch_size).copy()
    while psb.max() >= 8:
        j = 1 if kernel_sizes[-1][0] == 1 else 0
        psb[j:] = psb[j:] // 2
        kernel_sizes.append((1, 3, 3) if len(kernel_sizes) < n_2d_convs else (3, 3, 3))
        i += 1
    
    kernel_sizes_up = []
    for i in range(len(kernel_sizes) - 1):
        if kernel_sizes[i] == (1, 3, 3) and kernel_sizes[i+1] == (1, 3, 3):
            kernel_sizes_up.append((1, 3, 3))
        else:
            kernel_sizes_up.append((3, 3, 3))

    if use_prg_trn:
        
        assert isinstance(out_shape, list), "out_shapes must be given as a list of shapes introduced to the network in each stage"
        
        # padded for the augmentation
        prg_trn_sizes = [[s[0], 2*s[1], 2*s[2]] for s in out_shape]
        
        c = 4
        prg_trn_aug_params = {}
        prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.1/c], [0, 0.1]])
        prg_trn_aug_params['mm_sigma_blur'] = np.array([[0.5/c, 0.5 + 1/c], [0.5, 1.5]])
        prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.3/c, 1 + 0.3/c], [0.7, 1.3]])
        prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.35/c, 1 + 0.5/c], [0.65, 1.5]])
        prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 1/c], [1, 2]])
        prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.3/c, 1 + 0.5/c], [0.7, 1.5]])
        prg_trn_aug_params['out_shape'] = out_shape
        model_params['training']['prg_trn_sizes'] = prg_trn_sizes
        model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
        model_params['training']['prg_trn_resize_on_the_fly'] = False
    else:
        prg_trn_sizes = None

    model_params['training']['prg_trn_sizes'] = prg_trn_sizes
    model_params['network']['kernel_sizes'] = kernel_sizes
    model_params['network']['kernel_sizes_up'] = kernel_sizes_up
    model_params['network']['is_2d'] = False
    
    
    model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
    model_params['training']['lr_params'] = {'n_warmup_epochs': 50, 'lr_max': 0.02}
    model_params['training']['opt_params'] = {'momentum': 0.99,
                                              'weight_decay': 3e-5,
                                              'nesterov': True,
                                              'lr': 2*10**-2}

    padded_patch_size = patch_size.copy()
    padded_patch_size[1] = padded_patch_size[1] * 2
    padded_patch_size[2] = padded_patch_size[2] * 2

    model_params['augmentation']['torch_params']['grid_inplane']['out_shape'] = patch_size

    # next change the dataloader arguments
    for key in ['trn_dl_params', 'val_dl_params']:
        model_params['data'][key]['patch_size'] = patch_size
        model_params['data'][key]['batch_size'] = 2
        model_params['data'][key]['min_biased_samples'] = 1
        model_params['data'][key]['padded_patch_size'] = padded_patch_size
        model_params['data'][key]['bias'] = 'cl_fg'
        model_params['data'][key]['n_fg_classes'] = n_fg_classes

    model_params['prediction']['patch_size'] = patch_size

    return model_params

def get_model_params_3d_nfUNet(patch_size,
                               n_2d_convs,
                               use_prg_trn=False,
                               n_fg_classes=1,
                               fp32=False):
    model_params = get_model_params_3d_UNet(patch_size,
                                            n_2d_convs,
                                            use_prg_trn,
                                            n_fg_classes,
                                            fp32)
    model_params['architecture'] = 'nfUNet'
    del model_params['network']['norm']
    del model_params['network']['norm_params']
    del model_params['network']['kernel_sizes_up']
    model_params['network']['use_attention_gates'] = False
    model_params['network']['upsampling'] = 'conv'
    model_params['network']['align_corners']=True
    model_params['network']['factor_skip_conn']=1.0
    return model_params

def get_model_params_3d_res_encoder_U_Net(patch_size, z_to_xy_ratio, use_prg_trn=False,
                                          n_fg_classes=1, fp32=False, out_shape=None,
                                          larger_res_encoder=False):
    model_params = get_model_params_3d_UNet(patch_size, n_2d_convs=0, use_prg_trn=use_prg_trn,
                                            n_fg_classes=n_fg_classes, fp32=fp32, out_shape=out_shape)
    if out_shape is None and use_prg_trn:
        raise ValueError('Specify the out_shapes when using progressive training')
    
    del model_params['network']['kernel_sizes']
    del model_params['network']['kernel_sizes_up']
    del model_params['network']['n_pyramid_scales']
    model_params['architecture'] = 'unetresencoder'
    model_params['network']['block'] = 'res'
    model_params['network']['z_to_xy_ratio'] = z_to_xy_ratio
    model_params['network']['stochdepth_rate'] = 0
    if larger_res_encoder:
        model_params['network']['filters'] = 16
        model_params['network']['n_blocks_list'] = [1, 1, 2, 6, 3]
    else:
        model_params['network']['n_blocks_list'] = [1, 2, 6, 3]
    return model_params        

def get_model_params_effUNet(patch_size=[32, 256, 256],
                             z_to_xy_ratio=5/0.67,
                             use_prg_trn=True,
                             n_fg_classes=1,
                             out_shape=[[20, 160, 160], [24, 192, 192], [28, 224, 224], [32, 256, 256]],
                             larger_res_encoder=True):
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                         z_to_xy_ratio=z_to_xy_ratio,
                                                         use_prg_trn=use_prg_trn,
                                                         n_fg_classes=n_fg_classes,
                                                         out_shape=out_shape,
                                                         larger_res_encoder=larger_res_encoder,
                                                         fp32=False)
    model_params['architecture'] = 'unetresstemencoder'
    
    return model_params
    

def get_model_params_class_ensembling(prev_stages, patch_size, z_to_xy_ratio, n_fg_classes,
                                      use_prg_trn=False,
                                      fp32=False, out_shape=None,
                                      larger_res_encoder=False):
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size,z_to_xy_ratio, use_prg_trn,
                                                         n_fg_classes=n_fg_classes, fp32=fp32,
                                                         out_shape=out_shape,
                                                         larger_res_encoder=larger_res_encoder)
    if out_shape is None and use_prg_trn:
        raise ValueError('Specify the out_shapes when using progressive training')
    del model_params['training']['loss_params']
    model_params['network']['in_channels'] = 2
    model_params['prev_stages'] = prev_stages
    model_params['data']['folders'] = ['images', 'labels', 'bin_preds']
    model_params['data']['keys'] = ['image', 'label', 'bin_pred']
    model_params['data']['trn_dl_params']['pred_fps_key'] = 'bin_pred'
    # we set this to have only one foreground class here as the input is binary
    model_params['data']['trn_dl_params']['n_pred_classes'] = 1
    model_params['data']['val_dl_params']['pred_fps_key'] = 'bin_pred'
    model_params['data']['val_dl_params']['n_pred_classes'] = 1
    return model_params

# %%
def get_model_params_3d_cascade(prev_stage_preprocessed_name,
                                prev_stage_model_name,
                                patch_size,
                                n_2d_convs,
                                use_prg_trn=False,
                                n_fg_classes=1,
                                fp32=False):
    model_params = get_model_params_3d_UNet(patch_size=patch_size,
                                              n_2d_convs=n_2d_convs,
                                              use_prg_trn=use_prg_trn,
                                              n_fg_classes=n_fg_classes,
                                              fp32=fp32)
    model_params['augmentation']['np_params'] = {'mask': {'p_morph': 0.4,
                                                          'radius_mm': [1, 8],
                                                          'p_removal': 0.2,
                                                          'vol_percentage_removal': 0.15,
                                                          'vol_threshold_removal': None,
                                                          'threeD_morph_ops': False,
                                                          'aug_channels': [1]}}
    # account for additional inputs
    model_params['network']['in_channels'] = n_fg_classes + 1
    model_params['data']['keys'].append('pred_fps')
    model_params['data']['folders'].append(prev_stage_preprocessed_name + '_' +
                                           prev_stage_model_name)

    for dl in ['trn_dl_params', 'val_dl_params']:
        model_params['data'][dl]['pred_fps_key'] = 'pred_fps'
        model_params['data'][dl]['n_fg_classes'] = n_fg_classes

    model_params['prev_stage'] = {'preprocessed_name': prev_stage_preprocessed_name,
                                  'model_name': prev_stage_model_name}
    return model_params


# %%
def get_model_params_3d_from_preprocessed_folder(data_name,
                                                 preprocessed_name,
                                                 use_prg_trn=False,
                                                 fp32=False):

    path_to_params = os.path.join(os.environ['OV_DATA_BASE'],
                                  'preprocessed',
                                  data_name,
                                  preprocessed_name,
                                  'preprocessing_parameters.pkl')

    prep_params = pickle.load(open(path_to_params, 'rb'))

    if prep_params['apply_resizing']:
        spacing = prep_params['target_spacing']
    else:
        spacing = prep_params['dataset_properties']['median_spacing']

    spacing = np.array(spacing)
    if prep_params['apply_pooling']:
        spacing = spacing * np.array(prep_params['pooling_stride'])

    n_2d_convs = np.max([int(np.log2(spacing[0] / spacing[1]) + 0.5), 0])

    if n_2d_convs == 0:
        # isotropic case! The spacing in z direction is roughly as much as it is in xy
        patch_size = [96, 96, 96]
    elif n_2d_convs == 1:
        patch_size = [80, 160, 160]
    elif n_2d_convs == 2:
        patch_size = [48, 192, 192]
    elif n_2d_convs == 3:
        patch_size = [28, 224, 224]
    elif n_2d_convs == 4:
        patch_size = [20, 320, 320]
    else:
        raise NotImplementedError('It seems like your ')

    n_fg_classes = prep_params['dataset_properties']['n_fg_classes']

    return get_model_params_3d_UNet(patch_size, n_2d_convs, use_prg_trn=use_prg_trn,
                                      n_fg_classes=n_fg_classes, fp32=fp32)
