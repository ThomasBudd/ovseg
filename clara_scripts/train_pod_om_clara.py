from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
# add all the names of the labled training data sets as trn_data
parser.add_argument("trn_data", default='OV04', nargs='+',)
# vf should be 5,6,7. VF stands for validation folds. In this case
# the training is done on 100% of the data using 3 random seeds
parser.add_argument("vf", type=int)
args = parser.parse_args()

vf = args.vf
trn_data = args.trn_data
data_name = '_'.join(sorted(trn_data))
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

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     out_shape=out_shape,
                                                     n_fg_classes=2,
                                                     use_prg_trn=use_prg_trn)
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = wd

# change the model name when using other hyper-paramters
model_name = 'clara_model'

model = SegmentationModel(val_fold=vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
model.training.train()
if vf < model_params['data']['n_folds']:
    model.eval_validation_set()