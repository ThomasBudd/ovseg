from ovseg.model.model_parameters_segmentation import get_model_params_3d_cascade
from ovseg.model.SegmentationModel import SegmentationModel

model_params = get_model_params_3d_cascade(pred_fps_folder_name='test_test_test_cascade',
                                           patch_size=[16, 64, 64],
                                           n_2d_convs=2,
                                           fp32=True)

model_params['training']['num_epochs'] = 20
model_params['network']['filters'] = 8
model_params['data']['trn_dl_params']['epoch_len'] = 25
model_params['data']['val_dl_params']['epoch_len'] = 2

model = SegmentationModel(val_fold=0, data_name='Test', preprocessed_name='test_test',
                          model_name='test_cascade', model_parameters=model_params)
model.training.train()
model.eval_validation_set(save_preds=False)
