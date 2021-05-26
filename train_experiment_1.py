from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int)
# parser.add_argument("rep", type=int)
args = parser.parse_args()
p_name = 'pod_half'

weight_decay = 3e-4

# skip_type = "res_skip"
val_fold_list = list(range(5, 8))
exp_list = 3 * [args.gpu]


def get_model_params(exp):
    assert exp in [0, 1, 2, 3, 4, 5], "experiment must be 0 or 1"
    N = [1, 1, 1, 2, 2, 2][exp]
    M = [5, 10, 15, 5, 10, 15][exp]
    # M = [3, 4, 6, 3, 4, 6][exp]
    #weight_decay = [0, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4][exp]

    # model_name = 'weight_decay_{:.1e}'.format(weight_decay)
    model_name = 'myRandAugment_{}_{}'.format(N, M)
    patch_size = [32, 128, 128]
    prg_trn_sizes = [[16, 128, 128],
                     [24, 192, 192],
                     [32, 256, 256]]
    out_shape = [[16, 64, 64],
                 [24, 96, 96],
                 [32, 128, 128]]

    model_params = get_model_params_3d_nnUNet(patch_size, 2,
                                              use_prg_trn=True)
    del model_params['augmentation']['torch_params']['grayvalue']
    model_params['augmentation']['torch_params']['myRandAugment'] = {'n': N, 'm': M}
    model_params['training']['prg_trn_sizes'] = prg_trn_sizes
    
    # this time we change the amount of augmentation during training
    prg_trn_aug_params = {}
    c = 4
    prg_trn_aug_params['m'] = np.array([M/c, M])
    prg_trn_aug_params['out_shape'] = out_shape
    model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
    model_params['training']['prg_trn_resize_on_the_fly'] = False
    model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
    model_params['training']['lr_params'] = {'n_warmup_epochs': 50, 'lr_max': 0.02}
    model_params['training']['opt_params'] = {'momentum': 0.99, 'weight_decay': weight_decay,
                                              'nesterov': True,
                                              'lr': 0.02}
    model_params['training']['no_bias_weight_decay'] = True
    
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
