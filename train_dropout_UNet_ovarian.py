from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.data.Dataset import raw_Dataset
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

# delta_list = np.linspace(-2, 2, 25)[args.exp::5]

# delta_list = np.linspace(-3, 3, 37)[[args.exp, -1*args.exp]]

# delta = list(range(-3,4))[args.exp]

data_name = 'OV04'
preprocessed_name = 'pod_om_4fCV'

# equally scale down the patches to have ~half of the pixeld
patch_size = [32, 216, 216]
use_prg_trn = False
larger_res_encoder = False

wd = np.logspace(-4,-5,4)[args.exp]

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     n_fg_classes=2,
                                                     use_prg_trn=use_prg_trn)

model_params['data']['n_folds'] = 4
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = wd
model_params['training']['loss_params']['loss_names'] = ['dice_loss_sigm_weighted',
                                                         'cross_entropy_exp_weight']
    
    
w_list = [w_1, w_9]

model_params['training']['loss_params']['loss_kwargs'] = 2*[{'w_list':w_list}]
model_params['network']['p_dropout'] = 0.1
model_params['prediction']['mode'] = 'simple'
model_params['prediction']['use_training_mode_in_inference'] = True

model_name = f'dropout_UNet_{args.exp}'

model = SegmentationModel(val_fold=5,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
    
model.training.train()
ds_BARTS = raw_Dataset(os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'BARTS'))
ds_ApTC = raw_Dataset(os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'ApolloTCGA'))

for i in range(8):
    model.eval_ds(ds_BARTS, ds_name=f'BARTS_{i}', save_preds=True, save_plots=False,
                  force_evaluation=False)
    model.eval_ds(ds_ApTC, ds_name=f'ApolloTCGA_{i}', save_preds=True, save_plots=False,
                  force_evaluation=False)


