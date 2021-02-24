from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel

model_params = get_model_params_2d_segmentation()

val_fold = 0

data_name = 'kits19'
model_params['data']['n_folds'] = 3
model_params['data']['trn_dl_params']['store_coords_in_ram'] = False
model_params['data']['val_dl_params']['store_coords_in_ram'] = False

model_params['training']['num_epochs'] = 500
model_params['training']['lr_params']['lr_min'] = 0.5**0.9 * 0.01

model_pretrain = SegmentationModel(val_fold=0, data_name=data_name,
                                   model_name='pretrained_segmentation',
                                   model_parameters=model_params)
model_pretrain.training.train()
model_pretrain.validate()
