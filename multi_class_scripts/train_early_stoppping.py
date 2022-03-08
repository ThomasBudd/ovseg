from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.data.Dataset import raw_Dataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

patch_size = [32, 216, 216]
p_name = 'pod_om_08_5'
use_prg_trn = False
out_shape = None
larger_res_encoder = False


model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=2,
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                           'dice_loss']}

model_params['training']['stop_after_epochs'] = [250, 500, 750]
model_params['data']['val_dl_params']['n_fg_classes'] = 2
model_params['data']['trn_dl_params']['n_fg_classes'] = 2
model_params['data']['val_dl_params']['bias'] = 'cl_fg'
model_params['data']['trn_dl_params']['bias'] = 'cl_fg'

    
model = SegmentationModel(val_fold=0,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name='U-Net4_test_early_stopping',
                          model_parameters=model_params)


for i, ep in enumerate([250, 500, 750]):
    
    if model.training.epochs_done == ep:

        model.eval_ds(raw_Dataset('BARTS'),
                      'BARTS_{}'.format(ep),
                      save_preds=False)

while model.training.epochs_done < 1000:
    model.training.train()
    for i, ep in enumerate([250, 500, 750]):
        
        if model.training.epochs_done == ep:

            model.eval_ds(raw_Dataset('BARTS'),
                          'BARTS_{}'.format(ep),
                          save_preds=False)

model.eval_validation_set()
model.eval_raw_dataset('BARTS')