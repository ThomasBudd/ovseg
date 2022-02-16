from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("fold", type=int)
parser.add_argument("exp", type=int)
args = parser.parse_args()


if args.exp == 0:
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=[32, 216, 216],
                                                         z_to_xy_ratio=5.0/0.8,
                                                         use_prg_trn=False,
                                                         larger_res_encoder=False,
                                                         n_fg_classes=1)
    model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                              'dice_loss_weighted'],
                                              'loss_kwargs': [{'weight_bg': 1,
                                                               'n_fg_classes': 1},
                                                              {'eps': 1e-5,
                                                               'weight': 1}]}

    p_name = 'pod_08'
    model = SegmentationModel(val_fold=args.fold,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name='new_loss',
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set()
    model.eval_raw_data_npz('BARTS')
    model.clean()

    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=[32, 216, 216],
                                                         z_to_xy_ratio=5.0/0.8,
                                                         use_prg_trn=False,
                                                         larger_res_encoder=False,
                                                         n_fg_classes=4)
    model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                              'dice_loss_weighted'],
                                              'loss_kwargs': [{'weight_bg': 1,
                                                               'n_fg_classes': 4},
                                                              {'eps': 1e-5,
                                                               'weight': 1}]}
    p_name = 'new_multiclass_v1'
    model = SegmentationModel(val_fold=args.fold,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name='new_loss',
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set()
    model.eval_raw_data_npz('BARTS')
    model.clean()

    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=[32, 216, 216],
                                                         z_to_xy_ratio=5.0/0.8,
                                                         use_prg_trn=False,
                                                         larger_res_encoder=False,
                                                         n_fg_classes=7)
    model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                              'dice_loss_weighted'],
                                              'loss_kwargs': [{'weight_bg': 1,
                                                               'n_fg_classes': 7},
                                                              {'eps': 1e-5,
                                                               'weight': 1}]}
    p_name = 'new_multiclass_v2'
    model = SegmentationModel(val_fold=args.fold,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name='new_loss',
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set()
    model.eval_raw_data_npz('BARTS')
    model.clean()
else:
    for w in [0.01, 0.03, 0.3]:
        model_params = get_model_params_3d_res_encoder_U_Net([32, 216, 216],
                                                             5/0.8, 
                                                             n_fg_classes=2,
                                                             larger_res_encoder=False)
        
        model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                                  'dice_loss_weighted'],
                                                  'loss_kwargs': [{'weight_bg': w,
                                                                   'n_fg_classes': 2},
                                                                  {'eps': 1e-5,
                                                                   'weight': w}]}
        model_params['data']['folders'] = ['images', 'labels', 'regions']
        model_params['data']['keys'] = ['image', 'label', 'region']
        # we train using the regions as ground truht we're training for
        model_params['data']['trn_dl_params']['label_key'] = 'region'
        model_params['data']['val_dl_params']['label_key'] = 'region'
        
        model = RegionfindingModel(val_fold=args.fold,
                                   data_name='OV04',
                                   preprocessed_name='pod_om_reg',
                                   model_name='regfinding_{}'.format(w),
                                   model_parameters=model_params)
        
        model.training.train()
        model.eval_validation_set(save_preds=True, save_plots=False)
        model.eval_raw_data_npz('BARTS')