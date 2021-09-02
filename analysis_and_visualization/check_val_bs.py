from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
import numpy as np

model_name = 'debug'

patch_size = [32, 128, 128]
model_params = get_model_params_3d_nnUNet(patch_size, 2,
                                          use_prg_trn=False)
model_params['data']['val_dl_params']['batch_size'] = 2
model_params['data']['val_dl_params']['epoch_len'] = 8
model_params['data']['trn_dl_params']['epoch_len'] = 100
model_params['training']['num_epochs'] = 5
model_params['prediction']['batch_size'] = 25
p_name = 'pod_half'
val_fold = 0
model = SegmentationModel(val_fold=val_fold,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_raw_dataset('BARTS_test')