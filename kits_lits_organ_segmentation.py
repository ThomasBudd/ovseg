from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
import argparse
import numpy as np
import os
from ovseg import OV_PREPROCESSED

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
parser.add_argument("task", type=int)
args = parser.parse_args()
 
data_name = 'kits21_trn' if args.task == 0 else 'Lits19_trn'
preprocessed_name = 'organ'

if not os.path.exists(os.path.join(OV_PREPROCESSED, data_name, 'organ')):
    prep = SegmentationPreprocessingV2(apply_resizing=True, 
                                       apply_pooling=False, 
                                       apply_windowing=True,
                                       target_spacing=[5.0, 0.8, 0.8],
                                       lb_classes=[1])
    prep.plan_preprocessing_raw_data(data_name)

    prep.preprocess_raw_data(raw_data=data_name,
                             preprocessed_name=preprocessed_name)

wd = 1e-4
use_prg_trn = True

patch_size = [32, 216, 216]
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
z_to_xy_ratio = 5.0/0.8
larger_res_encoder = False
n_fg_classes = 1

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=z_to_xy_ratio,
                                                     out_shape=out_shape,
                                                     n_fg_classes=n_fg_classes,
                                                     use_prg_trn=use_prg_trn)
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = wd
model_params['data']['n_folds'] = 3

model_name = 'organ_segmentation'
model = SegmentationModel(val_fold=args.vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()

ens = SegmentationEnsemble(val_fold=list(range(3)),
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=preprocessed_name)
ens.wait_until_all_folds_complete()
ens.eval_raw_dataset(data_name.split('_')[0]+'_tst')
