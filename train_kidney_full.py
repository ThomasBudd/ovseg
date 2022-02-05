from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

vf = args.vf
data_name = 'kits21'
preprocessed_name = 'kidney_full'

# hyper-paramteres for the model
patch_size = [64, 256, 256]
use_prg_trn = True

sizes = 32*np.round(256 / np.arange(4,0,-1)**(1/3) / 32)

out_shape = [ [int(s)//4,int(s),int(s)] for s in sizes]
larger_res_encoder = True

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=4,
                                                     out_shape=out_shape,
                                                     n_fg_classes=1,
                                                     use_prg_trn=use_prg_trn)

model_name = 'larger_res_encoder'

model = SegmentationModel(val_fold=vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=4,
                                                     out_shape=out_shape,
                                                     n_fg_classes=1,
                                                     use_prg_trn=use_prg_trn)

model_params['architecture'] = 'UNet'
model_params['network']['kernel_sizes'] = [(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
model_params['network']['stem_kernel_size'] = [1, 2, 2]
model_params['n_pyramid_scales'] = 4

del model_params['network']['block']
del model_params['network']['z_to_xy_ratio']
del model_params['network']['n_blocks_list']
del model_params['network']['stochdepth_rate']

model_name = 'stem_unet'

model = SegmentationModel(val_fold=vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()
