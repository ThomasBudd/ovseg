from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel

model_params = get_model_params_2d_segmentation()

model_params['training']['num_epochs'] = 500
model_params['training']['lr_params']['lr_min'] = 0.5**0.9 * 0.01

model_pretrain = SegmentationModel(val_fold=0, data_name='all',
                                   model_name='pretrained_segmentation',
                                   model_parameters=model_params)
model_pretrain.training.train()
model_pretrain.validate()

model_params['training']['num_epochs'] = 1000
model_params['training']['lr_params']['lr_min'] = 0

model = SegmentationModel(val_fold=0, data_name='all', model_name='fully_trained_segmentation',
                          model_parameters=model_params)
model.training.load_last_checkpoint(model_pretrain.model_path)
model.training.train()
model.validate()
