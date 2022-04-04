from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

exp = args.exp
#raw_data = args.raw_data
data_name = 'ApolloTCGA_BARTS_OV04'
preprocessed_name = 'pod_om'

# hyper-paramteres for the model
wd = 1e-4
patch_size = [32, 216, 216]
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

weight = [1.0, 1.2, 1.4, 1.6, 1.8][exp]

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     out_shape=out_shape,
                                                     n_fg_classes=2,
                                                     use_prg_trn=use_prg_trn)
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = wd
model_params['training']['loss_params']['loss_names'] = ['cross_entropy_weighted_fg_v2', 
                                             'dice_loss_weighted']
model_params['training']['loss_params']['loss_kwargs'] = [{'weights_fg': [weight, weight]},
                                              {'weight': 2-weight}]

# change the model name when using other hyper-paramters
model_name = f'clara_model_weighted_{weight}'
for vf in range(5):
    model = SegmentationModel(val_fold=vf,
                              data_name=data_name,
                              model_name=model_name,
                              preprocessed_name=preprocessed_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set()
