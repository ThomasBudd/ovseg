from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

data_name = 'OV04'
preprocessed_name = 'pod_om_08_5'
i_process = args.i

wd = 1e-4

target_metrics = ['dice_1', 'dice_9']
validation_set_name = 'BARTS'

patch_size = [32, 216, 216]
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

n_epochs_per_stage = [1000, 1000]
vfs_per_stage = [[5], [6,7]]
n_processes = 10

mu = [0.98, 0.97][args.j]

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                             z_to_xy_ratio=5.0/0.8,
                                                             out_shape=out_shape,
                                                             n_fg_classes=2,
                                                             use_prg_trn=use_prg_trn)
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = wd

model_name = 'bs4'

model = SegmentationModel(val_fold=args.vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
model.training.train()
model.eval_training_set(save_preds=True)
model.eval_validation_set(save_preds=True)

