from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int)

args = parser.parse_args()
p_name = 'pod_half'

if args.gpu == 0:
    val_fold_list = [0, 4, 3, 2, 1]
    exp_list = [0, 0, 1, 2, 3]
elif args.gpu == 1:
    val_fold_list = [1, 0, 4, 3, 2]
    exp_list = [0, 1, 1, 2, 3]
elif args.gpu == 2:
    val_fold_list = [2, 1, 0, 4, 3]
    exp_list = [0, 1, 2, 2, 3]
elif args.gpu == 3:
    val_fold_list = [3, 2, 1, 0, 4]
    exp_list = [0, 1, 2, 3, 3]
elif args.gpu == 4:
    val_fold_list = list(range(5))
    exp_list = 5 * [4]


def get_model_params(exp):
    if exp == 0:
        model_name = 'add_prg_aug_update'
    elif exp == 1:
        model_name = 'prg_trn_5_stages'
    elif exp == 2:
        model_name = 'patch_size_40_192_5_stages'
    elif exp == 3:
        model_name = 'patch_size_40_160'
    elif exp == 4:
        model_name = 'patch_size_32_160'

    if exp <= 1:
        patch_size = [48, 192, 192]
    elif exp <= 2:
        patch_size = [40, 192, 192]
    elif exp == 3:
        patch_size = [40, 160, 160]
    else:
        patch_size = [32, 160, 160]

    model_params = get_model_params_3d_nnUNet(patch_size, 2,
                                              use_prg_trn=True)

    # this time we change the amount of augmentation during training
    prg_trn_aug_params = {}
    params = model_params['augmentation']['torch_params']['grayvalue']
    for key in params:
        if key.startswith('p'):
            prg_trn_aug_params[key] = [params[key]/2, params[key]]
        elif key.startswith('mm'):
            prg_trn_aug_params[key] = [np.array(params[key])/2, np.array(params[key])]
    params = model_params['augmentation']['torch_params']['grid_inplane']
    for key in params:
        if key.startswith('p'):
            prg_trn_aug_params[key] = [params[key]/2, params[key]]

    # in experiement 1 and 2 we will use 5 stages
    if exp == 1:
        model_params['training']['prg_trn_sizes'] = [[48, 64, 64], [48, 96, 96], [48, 128, 128],
                                                     [48, 160, 160], [48, 192, 192]]
    elif exp == 2:
        model_params['training']['prg_trn_sizes'] = [[40, 64, 64], [40, 96, 96], [40, 128, 128],
                                                     [40, 160, 160], [40, 192, 192]]
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
    del model.network
    for tpl in model.data.val_dl.dataset.data:
        for arr in tpl:
            del arr
        del tpl
    ens = SegmentationEnsemble(val_fold=list(range(5)),
                               data_name='OV04',
                               preprocessed_name=p_name,
                               model_name=model_name)
    if ens.all_folds_complete():
        ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)
