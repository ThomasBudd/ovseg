from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse

data_name = 'OV04'
preprocessed_name = 'pod_om'

# equally scale down the patches to have ~half of the pixeld
patch_size = [32, 216, 216]
use_prg_trn = True

out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     out_shape=out_shape,
                                                     n_fg_classes=2,
                                                     use_prg_trn=use_prg_trn)

model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = 1e-4
model_params['training']['loss_params']['loss_names'] = ['dice_loss_sigm_weighted',
                                                         'cross_entropy_exp_weight']
model_params['training']['stop_after_epochs'] = [750]
model_params['training']['loss_params']['loss_kwargs'] = 2*[{'w_list':[-1.5,-0.5]}]
# change the model name when using other hyper-paramters
model_name = 'stopped'

model = SegmentationModel(val_fold=5,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
model.training.train()