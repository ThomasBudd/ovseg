from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

vf = args.vf
data_name = 'kits21'
preprocessed_name = 'binary_quartered'

# hyper-paramteres for the model
patch_size = [96, 96, 96]
use_prg_trn = False

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=1,
                                                     n_fg_classes=1,
                                                     use_prg_trn=use_prg_trn)

model_params['architecture'] = 'UNet'
model_params['network']['kernel_sizes'] = 4*[(3, 3, 3)]
model_params['network']['norm'] = 'inst'
del model_params['network']['block']
del model_params['network']['z_to_xy_ratio']
del model_params['network']['n_blocks_list']
del model_params['network']['stochdepth_rate']

model_params['training']['num_epochs'] = 200
model_params['training']['opt_name'] = 'ADAM'
model_params['training']['opt_params'] = {'lr':0.001, 'betas': (0.99, 0.999)}
# change the model name when using other hyper-paramters
model_name = 'first_try'

model = SegmentationModel(val_fold=vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()
