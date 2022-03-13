from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

vf = args.vf
data_name = 'OV04'
preprocessed_name = 'pod_om_08_5'

# equally scale down the patches to have ~half of the pixeld
patch_size = [30, 192, 192]
use_prg_trn = True

sizes = 8*np.round(patch_size[1] / np.arange(4,0,-1)**(1/3) / 8)

out_shape = [ [int(s/patch_size[1]*patch_size[0]/2)*2,int(s),int(s)] for s in sizes]
larger_res_encoder = False

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     out_shape=out_shape,
                                                     n_fg_classes=2,
                                                     use_prg_trn=use_prg_trn)

# change the model name when using other hyper-paramters
model_name = 'smaller_patches_v1'

model = SegmentationModel(val_fold=vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
model.training.train()
model.eval_raw_data_npz('BARTS')
model.eval_raw_data_npz('ApolloTCGA')

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=preprocessed_name)
ens.wait_until_all_folds_complete()
ens.eval_raw_dataset('BARTS')
ens.eval_raw_dataset('ApolloTCGA')

patch_size = [22, 216, 216]
use_prg_trn = True

sizes = 8*np.round(patch_size[1] / np.arange(4,0,-1)**(1/3) / 8)

out_shape = [ [int(s/patch_size[1]*patch_size[0]/2)*2,int(s),int(s)] for s in sizes]
larger_res_encoder = False

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     out_shape=out_shape,
                                                     n_fg_classes=2,
                                                     use_prg_trn=use_prg_trn)

# change the model name when using other hyper-paramters
model_name = 'smaller_patches_v2'

model = SegmentationModel(val_fold=vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
model.training.train()
model.eval_raw_data_npz('BARTS')
model.eval_raw_data_npz('ApolloTCGA')

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=preprocessed_name)
ens.wait_until_all_folds_complete()
ens.eval_raw_dataset('BARTS')
ens.eval_raw_dataset('ApolloTCGA')