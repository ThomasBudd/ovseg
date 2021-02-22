from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel

model_params = get_model_params_2d_segmentation()

val_fold = 0

DEBUG = False

if DEBUG:
    data_name = 'Test'
    model_params['training']['num_epochs'] = 5
    model_params['data']['trn_dl_params']['batch_size'] = 2
    model_params['data']['trn_dl_params']['epoch_len'] = 25
    model_params['data']['val_dl_params']['batch_size'] = 2
    model_params['data']['val_dl_params']['epoch_len'] = 3
    model_params['network']['filters'] = 8
else:
    data_name = 'OV04'
    model_params['training']['num_epochs'] = 500
model_params['training']['lr_params']['lr_min'] = 0.5**0.9 * 0.01

model_pretrain = SegmentationModel(val_fold=0, data_name=data_name,
                                   model_name='pretrained_segmentation',
                                   model_parameters=model_params)
model_pretrain.training.train()
model_pretrain.validate()

if DEBUG:
    model_params['training']['num_epochs'] = 10
else:
    model_params['training']['num_epochs'] = 1000
model_params['training']['lr_params']['lr_min'] = 0

model = SegmentationModel(val_fold=0, data_name=data_name,
                          model_name='fully_trained_segmentation',
                          model_parameters=model_params)
model.training.load_last_checkpoint(model_pretrain.model_path)
model.training.train()
model.validate()
