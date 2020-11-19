import os


def get_model_params_2d_segmentation(aug_device='gpu', patch_size=[512, 512],
                                     n_fg_classes=1, n_stages=7,
                                     batch_size=12):
    model_parameters = {}
    # we're doing no preprocessing parameters here as they can be loaded
    # from the preprocessed folder

    # Let's jump straight to the augmentation
    grayvalue_params = {'p_noise': 0.15, 'var_noise_mm': [0, 0.1],
                        'p_blur': 0.1, 'sigma_blur_mm': [0.5, 1.5],
                        'blur_3d': False,
                        'p_bright': 0.15, 'fac_bright_mm': [0.7, 1.3],
                        'p_contr': 0.15, 'fac_contr_mm': [0.65, 1.5],
                        'p_gamma': 0.15, 'gamma_mm': [0.7, 1.5],
                        'p_gamma_inv': 0.15,
                        'aug_channels': [0]}
    spatial_params = {'patch_size': patch_size, 'p_scale': 0.2,
                      'scale_mm': [0.7, 1.4], 'p_rot': 0.2,
                      'rot_mm': [-180, 180], 'spatial_aug_3d': False,
                      'p_flip': 0.5, 'spacing': None, 'n_im_channels': 1}
    augmentation_params = {'grayvalue': grayvalue_params,
                           'spatial': spatial_params}
    model_parameters[aug_device+'_augmentation'] = augmentation_params

    # now the network parameters. classic 2d UNet
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
                     'epoch_len': 250, 'p_fg': 1/3, 'mn_fg': 1,
                     'padded_patch_size': None, 'memmap': 'r',
                     'store_coords_in_ram': True, 'num_workers': num_workers}
    val_dl_params = trn_dl_params.copy()
    val_dl_params['epoch_len'] = 25
    keys = ['image', 'label']
    folders = ['images', 'labels']
    data_params = {'n_folds': 5, 'fixed_shuffle': True, 'ds_params': ds_params,
                   'trn_dl_params': trn_dl_params, 'keys': keys,
                   'val_dl_params': val_dl_params, 'folders': folders}
    model_parameters['data'] = data_params

    # now finally the training!
    loss_params = {'eps': 1e-5, 'dice_weight': 0.5, 'pyramid_weight': 0.5}
    opt_params = {'momentum': 0.99, 'weight_decay': 3e-5, 'nesterov': True,
                  'lr': 10**-2}
    lr_params = {'beta': 0.9, 'lr_min': 10**-6}
    training_params = {'loss_params': loss_params,
                       'num_epochs': 1000, 'opt_params': opt_params,
                       'lr_params': lr_params, 'nu_ema_trn': 0.99,
                       'nu_ema_val': 0.7, 'fp32': False,
                       'p_plot_list': [0, 0.5, 0.8], 'opt_name': 'SGD'}
    model_parameters['training'] = training_params
    prediction_parameters = {'mode': 'flip'}
    model_parameters['prediction'] = prediction_parameters
    return model_parameters
