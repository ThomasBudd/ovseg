from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.data.Dataset import low_res_ds_wrapper
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()
patch_size = [32, 216, 216]
p_name = 'pod_om_08_5'
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

all_weights = [[1 + 0.05 * (i+1), 1] for i in range(6)] + [[1, 1 + 0.05 * (i+1)] for i in range(4)]
weights = all_weights[args.exp]

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=2,
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_fg',
                                                          'dice_loss_vector_weighted'],
                                           'loss_kwargs': [{'weights_fg': weights},
                                                           {'weights': weights}]}
model_params['training']['stop_after_epochs'] = [250, 500, 750]
model_params['data']['val_dl_params']['n_fg_classes'] = 2
model_params['data']['trn_dl_params']['n_fg_classes'] = 2
model_params['data']['val_dl_params']['bias'] = 'cl_fg'
model_params['data']['trn_dl_params']['bias'] = 'cl_fg'

    
model = SegmentationModel(val_fold=0,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name='U-Net4_weighted_{:.2f}_{:.2f}'.format(*weights),
                          model_parameters=model_params)

while model.training.epochs_done < 1000:
    model.training.train()
    for i, ep in enumerate([250, 500, 750]):
        
        if model.training.epochs_done == ep:
    
            scale = (np.array(out_shape[i]) / np.array(out_shape[-1])).tolist()
            BARTS_low_res_ds = low_res_ds_wrapper('BARTS', scale)
            # the model has been trained on this patch size so far
            # we have to also use it for inference
            model.prediction.patch_size = out_shape[i]
            model.eval_ds(BARTS_low_res_ds, 'BARTS_{}'.format(ep),
                          save_preds=False)
            # undo this before bad things happen
            model.prediction.patch_size = patch_size

model.eval_raw_dataset('BARTS')