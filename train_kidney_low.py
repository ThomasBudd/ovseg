from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

vf = args.vf
data_name = 'kits21'
preprocessed_name = 'kidney_low'

# hyper-paramteres for the model
wd = 1e-4
patch_size = [96, 96, 96]
use_prg_trn = True

sizes = 8*np.ceil(96 / np.arange(4,0,-1)**(1/3) / 8)

out_shape = [ 3*[int(s)] for s in sizes]
larger_res_encoder = False

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=1,
                                                     out_shape=out_shape,
                                                     n_fg_classes=1,
                                                     use_prg_trn=use_prg_trn)
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = wd
# change the model name when using other hyper-paramters
model_name = 'first_try'

model = SegmentationModel(val_fold=vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()
