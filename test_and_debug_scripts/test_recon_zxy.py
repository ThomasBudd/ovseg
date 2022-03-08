from ovseg.networks.recon_networks import get_operator
from ovseg.preprocessing.Reconstruction2dSimPreprocessing import Reconstruction2dSimPreprocessing
from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
import os

op = get_operator(256)
prep = Reconstruction2dSimPreprocessing(op, num_photons= 2*10**6)

if not os.path.exists(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed',
                                   'OV04', 'pod_full', 'projections')):
    prep.preprocess_raw_folders('OV04', preprocessed_name='pod_full')


model_params = get_model_params_2d_reconstruction('post_processing_UNet')
model_params['data']['trn_dl_params']['store_data_in_ram'] = True
model_params['training']['num_epochs'] = 1000
model = Reconstruction2dSimModel(val_fold=1, data_name='OV04', model_name='post_processing_UNet_1000',
                                 model_parameters=model_params, preprocessed_name='pod_full')

model.training.train()
model.eval_validation_set(save_preds=False, save_plots=False)
