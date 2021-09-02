from ovseg.model.model_parameters_reconstruction import get_model_params_2d_reconstruction
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("i")

args = parser.parse_args()

if int(args.i) == 0:
    batch_size = 8
    num_epochs_list = [500, 750, 1000]
elif int(args.i) == 1:
    batch_size = 12
    num_epochs_list = [250, 375, 500]

for num_epochs in num_epochs_list:
    
    model_name = 'post_processing_UNet_{}_{}'.format(batch_size, num_epochs)
    
    model_params = get_model_params_2d_reconstruction('post_processing_UNet')
    model_params['data']['trn_dl_params']['store_data_in_ram'] = True
    model_params['data']['trn_dl_params']['batch_size'] = batch_size
    model_params['data']['val_dl_params']['batch_size'] = batch_size
    model_params['training']['num_epochs'] = num_epochs
    model = Reconstruction2dSimModel(val_fold=1, data_name='OV04', model_name='post_processing_UNet_1000',
                                     model_parameters=model_params, preprocessed_name='pod_full')
    
    model.training.train()
    model.eval_validation_set(save_preds=False, save_plots=False)
