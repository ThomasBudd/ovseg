from ovseg.model.ImageProcessingModel import ImageProcessingModel
import os
from ovseg.model.model_parameters_imageprocessing import get_model_params_2d_imageprocessing

model_params = get_model_params_2d_imageprocessing()

model_params['data']['trn_dl_params']['batch_size'] = 3
model_params['data']['val_dl_params']['batch_size'] = 3

if not os.path.exists(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04',
                                   'pod_default', 'images_win_norm')):
    from ovseg.preprocessing.ImageProcessingPreprocessing import ImageProcessingPreprocessing
    preprocessing = ImageProcessingPreprocessing(**model_params['preprocessing'])
    preprocessing.preprocess_raw_folders(['OV04', 'BARTS', 'ApolloTCGA'],
                                         'pod_default',
                                         data_name='OV04')

model = ImageProcessingModel(val_fold=0, data_name='OV04', model_parameters=model_params,
                             model_name='identity_preprocessing')

model.training.tain()
model.eval_validation_set(save_preds=False, plot=False)
model.eval_training_set(save_preds=False, plot=False)
