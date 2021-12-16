from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net

patch_size = [32, 216, 216]
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

data_name = 'OV04'
p_name = 'pod_om_08_5'

model_name = 'debug_focal_loss'


model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5/0.8,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=2,
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                          'dice_loss']}
model_params['training']['loss_params'] = {'loss_names': ['unifiedFocalLoss'],
                                           'loss_kwargs': {'delta': 0.5,
                                                           'gamma': 0.4,
                                                           'eps': 1e-5}} 
    
model = SegmentationModel(val_fold=0,
                          data_name=data_name,
                          preprocessed_name=p_name, 
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()