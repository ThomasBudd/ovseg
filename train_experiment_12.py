from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_cascade
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("p", type=int)
args = parser.parse_args()

p_name = 'pod_full'

# skip_type = "res_skip"
val_fold_list = [9, 9]
exp_list = [[0, 1], [2, 3], [4]][args.p]


def get_model_params(exp):
    # model_name = 'weight_decay_{:.1e}'.format(weight_decay)
    model_name = ['res_refine', 'res_refine_no_gv_augs', 'label_aug_rare', 'label_aug_freq',
                  'res_refine_500'][exp]

    patch_size = [20, 160, 160]
    model_params = get_model_params_3d_cascade('pod_half',
                                               'res_encoder_p_bias_0.5',
                                               patch_size,
                                               n_2d_convs=3,
                                               use_prg_trn=False)

    del model_params['network']
    model_params['architecture'] = 'refineresnet'
    model_params['data']['trn_dl_params']['min_biased_samples'] = 0
    model_params['data']['val_dl_params']['min_biased_samples'] = 0
    model_params['data']['trn_dl_params']['p_bias_sampling'] = 0.5
    model_params['data']['val_dl_params']['p_bias_sampling'] = 0.5
    model_params['data']['trn_dl_params']['bias']='mv'
    model_params['data']['val_dl_params']['bias']='mv'
    model_params['network']= {'in_channels': 2, 'out_channels': 2,'hid_channels': 32, 
                              'z_to_xy_ratio': 8}

    if exp == 1:
        del model_params['augmentation']['torch_params']['grayvalue']

    if exp == 2:
        model_params['augmentation']['np_params'] = {'mask': {'p_morph': 0.4,
                                                              'radius_mm': [0, 2],
                                                              'p_removal': 0.0,
                                                              'vol_percentage_removal': 0.15,
                                                              'vol_threshold_removal': None,
                                                              'threeD_morph_ops': False,
                                                              'aug_channels': [1]}}
    elif exp == 3:
        model_params['augmentation']['np_params'] = {'mask': {'p_morph': 1.0,
                                                              'radius_mm': [0, 2],
                                                              'p_removal': 0.0,
                                                              'vol_percentage_removal': 0.15,
                                                              'vol_threshold_removal': None,
                                                              'threeD_morph_ops': False,
                                                              'aug_channels': [1]}}
    else:
        del model_params['augmentation']['np_params']
    if exp == 4:
        model_params['training']['num_epochs'] = 500
        n_warmup_epochs = 25
    else:
        model_params['training']['num_epochs'] = 250
        n_warmup_epochs = 13
        

    model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
    model_params['training']['lr_params'] = {'n_warmup_epochs': n_warmup_epochs,
                                             'lr_max': 3 * 10**-4}
    model_params['training']['opt_name'] = 'Adam'
    model_params['training']['opt_params'] = {'lr': 3 * 10**-4}
    return model_params, model_name


for val_fold, exp in zip(val_fold_list, exp_list):
    model_params, model_name = get_model_params(exp)
    model = SegmentationModel(val_fold=val_fold,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name=model_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_training_set()
    model.eval_raw_data_npz('BARTS')
    model.clean()
    
    # if val_fold == 3:
    #     ens = SegmentationEnsemble(val_fold=list(range(5)),
    #                                data_name='OV04',
    #                                preprocessed_name=p_name,
    #                                model_name=model_name)
    #     if ens.all_folds_complete():
    #         ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)
