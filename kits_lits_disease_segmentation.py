from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.SegmentationEnsembleV2 import SegmentationEnsembleV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
import argparse
import numpy as np
import os
from ovseg import OV_PREPROCESSED

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
parser.add_argument("task", type=int)
args = parser.parse_args()
 
data_name = 'kits21_trn' if args.task == 0 else 'Lits19_trn'
preprocessed_name = 'disease'

if not os.path.exists(os.path.join(OV_PREPROCESSED, data_name, preprocessed_name)):
    lb_classes = [2,3] if args.task == 0 else [2]
    
    prep = SegmentationPreprocessingV2(apply_resizing=True, 
                                       apply_pooling=False, 
                                       apply_windowing=True,
                                       target_spacing=[5.0, 0.8, 0.8],
                                       lb_classes=lb_classes,
                                       prev_stage_for_mask={'data_name':data_name,
                                                            'preprocessed_name': 'organ',
                                                            'model_name': 'organ_segmentation'})
    prep.plan_preprocessing_raw_data(data_name)

    prep.preprocess_raw_data(raw_data=data_name,
                             preprocessed_name=preprocessed_name)

wd = 1e-4
use_prg_trn = True

patch_size = [32, 216, 216]
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
z_to_xy_ratio = 5.0/0.8
larger_res_encoder = False
n_fg_classes = 2 if args.task == 0 else 1

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=z_to_xy_ratio,
                                                     out_shape=out_shape,
                                                     n_fg_classes=n_fg_classes,
                                                     use_prg_trn=use_prg_trn)
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = wd
model_params['data']['n_folds'] = 3
model_params['training']['loss_params']['loss_names'] = ['dice_loss_sigm_weighted',
                                                         'cross_entropy_exp_weight']
w_list = [0, 0] if args.task == 0 else [0]
model_params['training']['loss_params']['loss_kwargs'] = 2*[{'w_list':w_list}]
model_params['training']['stop_after_epochs'] = [750]
model_name = 'stopped'
model = SegmentationModelV2(val_fold=args.vf,
                            data_name=data_name,
                            model_name=model_name,
                            preprocessed_name=preprocessed_name,
                            model_parameters=model_params)
model.training.train()

for w in list(range(-2,3)):
    
    w_list = [w, w] if args.task == 0 else [w]
    
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                         z_to_xy_ratio=5.0/0.8,
                                                         n_fg_classes=2,
                                                         use_prg_trn=False)
    
    model_params['data']['n_folds'] = 3
    model_params['data']['trn_dl_params']['batch_size'] = 4
    model_params['data']['val_dl_params']['batch_size'] = 4
    model_params['training']['opt_params']['momentum'] = 0.98
    model_params['training']['opt_params']['weight_decay'] = 1e-4
    model_params['training']['loss_params']['loss_names'] = ['dice_loss_sigm_weighted',
                                                             'cross_entropy_exp_weight']
    model_params['training']['loss_params']['loss_kwargs'] = 2*[{'w_list':w_list}]
    
    model = SegmentationModelV2(val_fold=args.vf,
                              data_name=data_name,
                              model_name=model_name,
                              preprocessed_name=preprocessed_name,
                              model_parameters=model_params)
    path_to_checkpoint = os.path.join(os.environ['OV_DATA_BASE'],
                                        'trained_models',
                                        data_name,
                                        preprocessed_name,
                                        'stopped',
                                        f'fold_{args.vf}')
    model.training.load_last_checkpoint(path_to_checkpoint)
    model.training.loss_params = {'loss_names': ['dice_loss_sigm_weighted',
                                                 'cross_entropy_exp_weight'],
                                  'loss_kwargs': 2*[{'w_list':w_list}]}
    model.training.initialise_loss()
    model.training.save_checkpoint()
    model.training.train()
    model.eval_validation_set()
