from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
# parser.add_argument("vf", type=int)
parser.add_argument("exp", type=int)
args = parser.parse_args()

# vf = args.vf

w_1 = -1.5
w_9 = -0.5

delta_list = np.linspace(-2, 2, 25)[args.exp::5]

data_name = 'OV04'
preprocessed_name = 'pod_om_4fCV'

# equally scale down the patches to have ~half of the pixeld
patch_size = [32, 216, 216]
use_prg_trn = False
larger_res_encoder = False

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     n_fg_classes=2,
                                                     use_prg_trn=use_prg_trn)

model_params['data']['n_folds'] = 4
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = 1e-4
model_params['training']['loss_params']['loss_names'] = ['dice_loss_sigm_weighted',
                                                         'cross_entropy_exp_weight']
    
# change the model name when using other hyper-paramters

for delta in delta_list:
    
    w_list = [w_1 + delta, w_9 + delta]
    
    model_params['training']['loss_params']['loss_kwargs'] = 2*[{'w_list':w_list}]
    
    model_name = f'calibrated_{delta:.2f}'
    
    model = SegmentationModel(val_fold=5,
                              data_name=data_name,
                              model_name=model_name,
                              preprocessed_name=preprocessed_name,
                              model_parameters=model_params)
    
    path_to_checkpoint = os.path.join(os.environ['OV_DATA_BASE'],
                                      'trained_models',
                                      data_name,
                                      preprocessed_name,
                                      model_name,
                                      f'fold_{vf}',
                                      'attribute_checkpoint.pkl')
    if os.path.exists(path_to_checkpoint):
        print('Previous checkpoint found and loaded')
    else:
        print('Loading pretrained checkpoint')
        path_to_checkpoint = os.path.join(os.environ['OV_DATA_BASE'],
                                          'trained_models',
                                          data_name,
                                          preprocessed_name,
                                          'stopped',
                                          'fold_5')
        model.training.load_last_checkpoint(path_to_checkpoint)
        model.training.loss_params = {'loss_names': ['dice_loss_sigm_weighted',
                                                     'cross_entropy_exp_weight'],
                                      'loss_kwargs': 2*[{'w_list':w_list}]}
        model.training.initialise_loss()
        model.training.save_checkpoint()
    model.training.train()
    model.eval_raw_dataset('BARTS')
    model.eval_raw_dataset('ApolloTCGA')