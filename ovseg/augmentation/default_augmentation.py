from ovseg.augmentation import SegmentationAugmentation


def get_default_gv_augm_params(is_3d: bool):
    params = {'p_noise': 0.15, 'var_noise_mm': [0, 0.1], 'p_blur': 0.1,
              'sigma_blur_mm': [0.5, 1.5], 'p_bright': 0.15,
              'fac_bright_mm': [0.7, 1.3], 'p_contr': 0.15,
              'fac_contr_mm': [0.65, 1.5], 'p_gamma': 0.15,
              'gamma_mm': [0.7, 1.5], 'p_gamma_inv': 0.15, 'aug_channels': [0],
              'blur_3d': is_3d}
    return params


def get_default_spatial_augm_params(patch_size, is_3d: bool, spacing=None):
    params = {'p_scale': 0.2, 'scale_mm': [0.7, 1.4], 'p_rot': 0.2,
              'rot_mm': [-15, 15], 'spatial_aug_3d': is_3d, 'p_flip': 0.5,
              'spacing': spacing, 'patch_size': patch_size}
    return params


def get_default_augm_params(patch_size, is_3d: bool, spacing=None):
    augm_params = {'GrayValueAugmentation': get_default_gv_augm_params(is_3d),
                   'SpatialAugmentation':
                       get_default_spatial_augm_params(patch_size, is_3d,
                                                       spacing)}
    return augm_params


def get_default_augmentation(patch_size, is_3d: bool, spacing=None):
    augmentation_params = get_default_augm_params(patch_size, is_3d, spacing)
    return SegmentationAugmentation.\
        SegmentationAugmentation(augmentation_params)
