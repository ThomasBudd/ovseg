from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
from ovseg.preprocessing.Reconstruction2dSimPreprocessing import Reconstruction2dSimPreprocessing
from ovseg.network.recon_networks import get_operator
import os


model_params = get_model_params_2d_reconstruction(image_folder='images_HU_rescale',
                                                  projection_folder='projections_HU')
model_params['preprocessing']['window'] = None

data_name = 'OV04'
preprocessed_name = 'pod_default'
val_fold = 0
if not os.path.exists(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', data_name,
                                   preprocessed_name, 'images_HU_rescale')):
    op = get_operator()
    preprocessing = Reconstruction2dSimPreprocessing(op, **model_params['preprocessing'])
    preprocessing.preprocess_raw_folders(['OV04', 'BARTS', 'ApolloTCGA'],
                                         preprocessed_name=preprocessed_name,
                                         data_name=data_name,
                                         proj_folder_name='projections_HU',
                                         im_folder_name='images_HU_rescale')

model = Reconstruction2dSimModel(val_fold=val_fold,
                                 data_name=data_name,
                                 preprocessed_name=preprocessed_name,
                                 model_parameters=model_params,
                                 model_name='recon_fbp_convs_full_HU',
                                 plot_window=[-50, 350])
model.training.train()

model.eval_validation_set(save_preds=True)
model.eval_training_set(save_preds=True)
