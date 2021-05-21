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
    val_fold_list = [5, 5]
    exp_list = [0, 2]
elif args.gpu == 1:
    val_fold_list = [6, 6]
    exp_list = [0, 2]
elif args.gpu == 2:
    val_fold_list = [7, 7]
    exp_list = [0, 2]
elif args.gpu == 3:
    val_fold_list = [5, 6]
    exp_list = [1, 1]
elif args.gpu == 4:
    val_fold_list = [7]
    exp_list = [1]


def get_model_params(exp):
    if exp == 0:
        model_name = '3d_prg_lrn_48_192'
        patch_size = [48, 192, 192]
        prg_trn_sizes = [[24, 192, 192],
                         [32, 256, 256],
                         [40, 320, 320],
                         [48, 384, 384]]
        out_shape = [[24, 96, 96],
                     [32, 128, 128],
                     [40, 160, 160],
                     [48, 192, 192]]
        
    elif exp == 1:
        model_name = '3d_prg_lrn_40_160'
        patch_size = [40, 160, 160]
        prg_trn_sizes = [[16, 128, 128],
                         [24, 192, 192],
                         [32, 256, 256],
                         [40, 320, 320]]
        out_shape = [[16, 64, 64],
                     [24, 96, 96],
                     [32, 128, 128],
                     [40, 160, 160]]
    elif exp == 2:
        model_name = '3d_prg_lrn_32_128'
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
    ens = SegmentationEnsemble(val_fold=list(range(5, 8)),
                               data_name='OV04',
                               preprocessed_name=p_name,
                               model_name=model_name)
    if ens.all_folds_complete():
        ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)