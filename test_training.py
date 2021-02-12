from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation

model_params = get_model_params_2d_segmentation()

model_params['data']['trn_dl_params']['batch_size'] = 3
model_params['data']['trn_dl_params']['epoch_len'] = 25
model_params['data']['val_dl_params']['batch_size'] = 3
model_params['data']['val_dl_params']['epoch_len'] = 2
model_params['training']['num_epochs'] = 50
model_params['network']['filters'] = 8

model = SegmentationModel(val_fold=1, data_name='OV04', preprocessed_name='default',
                          model_parameters=model_params, model_name='test_training',
                          dont_store_data_in_ram=True)

model.training.train()
model.eval_validation_set()
model.eval_training_set()
