from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int)
parser.add_argument("rep", type=int)
args = parser.parse_args()
p_name = 'pod_half'

# skip_type = "res_skip"
val_fold_list = list(range(5, 8))
exp_list = 3 * [args.gpu]


def get_model_params(exp):
    assert exp in [0, 1, 2, 3], "experiment must be 0 or 1"
    if exp == 0:
        use_trilinear_upsampling=True
        use_less_hid_channels_in_decoder=False
        fac_skip_channels=1
    elif exp == 1:
        use_trilinear_upsampling=False
        use_less_hid_channels_in_decoder=False
        fac_skip_channels=0.5
    elif exp == 2:
        use_trilinear_upsampling=True
        use_less_hid_channels_in_decoder=True
        fac_skip_channels=0.5
    else:
        use_trilinear_upsampling=True
        use_less_hid_channels_in_decoder=False
        fac_skip_channels=0.5

    model_name = 'decoder'
    if use_trilinear_upsampling:
        model_name += '_tril'
    if use_less_hid_channels_in_decoder:
        model_name += '_<hidch'
    if fac_skip_channels == 0.5:
        model_name += '_skip_0.5'
    model_name += '_{}'.format(args.rep)
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
    model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
    model_params['training']['prg_trn_resize_on_the_fly'] = False
    model_params['network']['use_trilinear_upsampling'] = use_trilinear_upsampling
    model_params['network']['use_less_hid_channels_in_decoder'] = use_less_hid_channels_in_decoder
    model_params['network']['fac_skip_channels'] = fac_skip_channels
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

ens = SegmentationEnsemble(val_fold=list(range(5, 8)),
                           data_name='OV04',
                           preprocessed_name=p_name,
                           model_name=model_name)
if ens.all_folds_complete():
    ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)

# %%
import torch
for batch in model.data.trn_dl:
    break

xb = batch.cuda().type(torch.float)[:, :1, :, :128, :128]
self = model.network
