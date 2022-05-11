from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
# vf should be 5,6,7. VF stands for validation folds. In this case
# the training is done on 100% of the data using 3 random seeds
parser.add_argument("vf", type=int)
# this should be either of ['pod_om', 'abdominal_lesions','lymph_nodes']
# to indicate which of the three segmentation models is trained
parser.add_argument("--model", default='pod_om')
# add all the names of the labled training data sets as trn_data
parser.add_argument("--trn_data", default=['OV04', 'BARTS', 'ApolloTCGA'], nargs='+')
args = parser.parse_args()

vf = args.vf
trn_data = args.trn_data
data_name = '_'.join(sorted(trn_data))
preprocessed_name = args.model

assert preprocessed_name in ['pod_om', 'abdominal_lesions','lymph_nodes'], 'Unkown model'

# hyper-paramteres for the model
wd = 1e-4
use_prg_trn = True

if preprocessed_name == 'pod_om':
    patch_size = [32, 216, 216]
    out_shape = [[20, 128, 128],
                 [22, 152, 152],
                 [30, 192, 192],
                 [32, 216, 216]]
    z_to_xy_ratio = 5.0/0.8
    larger_res_encoder = False
    n_fg_classes = 2
elif preprocessed_name == 'abdominal_lesions':
    patch_size = [32, 216, 216]
    out_shape = [[20, 128, 128],
                 [22, 152, 152],
                 [30, 192, 192],
                 [32, 216, 216]]
    z_to_xy_ratio = 5.0/0.8
    larger_res_encoder = False
    n_fg_classes = 6
elif preprocessed_name == 'lymph_nodes':
    patch_size = [32, 256, 256]
    use_prg_trn = True
    out_shape = [[20, 160, 160],
                 [24, 192, 192],
                 [28, 224, 224],
                 [32, 256, 256]]
    z_to_xy_ratio = 5.0/0.67
    larger_res_encoder = True
    n_fg_classes = 4

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=z_to_xy_ratio,
                                                     out_shape=out_shape,
                                                     n_fg_classes=n_fg_classes,
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