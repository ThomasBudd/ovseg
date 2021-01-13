import os


def get_model_params_2d_segmentation(aug_device='gpu', patch_size=[512, 512],
                                     n_fg_classes=1, n_stages=7,
                                     batch_size=12, pred_mode='flip'):
    model_parameters = {}
    # we're doing no preprocessing parameters here as they can be loaded
    # from the preprocessed folder

    # Let's jump straight to the augmentation
    grayvalue_params = {'p_noise': 0.15, 'var_noise_mm': [0, 0.1],
                        'p_blur': 0.1, 'sigma_blur_mm': [0.5, 1.5],
                        'blur_3d': False,
                        'p_bright': 0.15, 'fac_bright_mm': [0.7, 1.3],
                        'p_contr': 0.15, 'fac_contr_mm': [0.65, 1.5],
                        'p_gamma': 0.0, 'gamma_mm': [0.7, 1.5],
                        'p_gamma_inv': 0.0,
                        'aug_channels': [0]}
    spatial_params = {'patch_size': patch_size, 'p_scale': 0.2,
                      'scale_mm': [0.7, 1.4], 'p_rot': 0.2,
                      'rot_mm': [-180, 180], 'spatial_aug_3d': False,
                      'p_flip': 0.5, 'spacing': None, 'n_im_channels': 1}

    augmentation_params = {'grayvalue': grayvalue_params,
                           'spatial': spatial_params}

    model_parameters['augmentation'] = {'GPU_params': None, 'CPU_params': None, 'TTA_params': None}
    if aug_device == 'gpu':
        model_parameters['augmentation']['GPU_params'] = augmentation_params
    elif aug_device == 'cpu':
        model_parameters['augmentation']['CPU_params'] = augmentation_params
    else:
        raise ValueError('aug_device must be \'gpu\' or \'cpu\'.')

    if pred_mode.lower() == 'tta':
        model_parameters['augmentation']['TTA_params'] = spatial_params

    # now the network parameters. classic 2d UNet
    model_parameters['architecture'] = 'UNet'
    network_parameters = {'in_channels': 1, 'out_channels': n_fg_classes+1,
                          'kernel_sizes': n_stages*[3],
                          'is_2d': True, 'filters': 32,
                          'filters_max': 384, 'n_pyramid_scales': None}
    model_parameters['network'] = network_parameters

    # we have no postprocessing parameters and just go with the default
    # now data
    ds_params = {}
    num_workers = 0 if os.name == 'nt' else 8
    trn_dl_params = {'patch_size': patch_size, 'batch_size': batch_size,
                     'epoch_len': 250, 'p_fg': 0, 'mn_fg': 3,
                     'padded_patch_size': None, 'memmap': 'r',
                     'store_coords_in_ram': True, 'num_workers': num_workers}
    val_dl_params = trn_dl_params.copy()
    val_dl_params['epoch_len'] = 25
    keys = ['image', 'label']
    folders = ['images', 'labels']
    data_params = {'n_folds': 4, 'fixed_shuffle': True, 'ds_params': ds_params,
                   'trn_dl_params': trn_dl_params, 'keys': keys,
                   'val_dl_params': val_dl_params, 'folders': folders}
    model_parameters['data'] = data_params

    # prediction object
    prediction_params = {'batch_size': 1, 'overlap': 0.5, 'fp32': False,
                         'patch_weight_type': 'constant', 'sigma_gaussian_weight': 1,
                         'mode': pred_mode, 'TTA_n_full_predictions': 1, 'TTA_n_max_augs': 99,
                         'TTA_eps_stop': 0.02}
    model_parameters['prediction'] = prediction_params

    # now finally the training!
    loss_params = {'eps': 1e-5,
                   'dice_weight': 1.0,
                   'ce_weight': 1.0,
                   'pyramid_weight': 0.5}
    opt_params = {'momentum': 0.99, 'weight_decay': 3e-5, 'nesterov': True,
                  'lr': 10**-2}
    lr_params = {'beta': 0.9, 'lr_min': 0}
    training_params = {'loss_params': loss_params,
                       'num_epochs': 1000, 'opt_params': opt_params,
                       'lr_params': lr_params, 'nu_ema_trn': 0.99,
                       'nu_ema_val': 0.7, 'fp32': False,
                       'p_plot_list': [1, 0.5, 0.2], 'opt_name': 'SGD'}
    model_parameters['training'] = training_params
    return model_parameters


def get_model_params_iUNet_segmentation(patch_size_in=[256, 256, 32],
                                        patch_size_aug=[320, 320, 32]):
    model_params = get_model_params_2d_segmentation(batch_size=2)
    model_params['augmentation']['GPU_params']['spatial']['patch_size'] = patch_size_in
    model_params['architecture'] = 'iUNet'
    for s in ['val_dl_params', 'trn_dl_params']:
        model_params['data'][s]['patch_size'] = patch_size_aug
        model_params['data'][s]['mn_fg'] = 1

    # EDIT DEFAULT VALUES FOR NETWORK HERE
    model_params['network'] = {}
    return model_params
