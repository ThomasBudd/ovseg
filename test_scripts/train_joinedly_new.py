from ovseg.data.JoinedData import JoinedData
import os
from ovseg.training.JoinedTraining import JoinedTraining
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
from ovseg.model.SegmentationModel import SegmentationModel
import pickle
import numpy as np
import matplotlib.pyplot as plt

weights = [0.1, 0.5, 0.9, 1]
doses = ['full', 'quater']

# for dose in doses:
    # for loss_weight in weights:

dose = 'full'
for loss_weight in [0.5, 0.9, 1.0]:
    val_fold = 0
    data_name = 'all'
    
    trn_dl_params = {'batch_size': 12, 'patch_size': [512, 512],
                     'num_workers': None, 'pin_memory': True,
                     'epoch_len': 250, 'store_coords_in_ram': False}
    val_dl_params = {'batch_size': 12, 'patch_size': [512, 512],
                     'num_workers': None, 'pin_memory': True,
                     'epoch_len': 25, 'store_coords_in_ram': False}
    
    preprocessed_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed',
                                     data_name, 'default')
    keys = ['projection', 'image', 'label', 'spacing', 'orig_shape']
    folders = ['projections_'+dose, 'images_att', 'labels', 'spacings',
               'orig_shapes']
    data = JoinedData(val_fold, preprocessed_path, keys, folders,
                      trn_dl_params=trn_dl_params,
                      val_dl_params=val_dl_params)
    # %% load models
    model1 = Reconstruction2dSimModel(val_fold, data_name, 'reconstruction_'+dose)
    model_path = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models',
                              data_name, 'warm_start_no_gamma')
    model_params = pickle.load(open(os.path.join(model_path,
                                                 'model_parameters.pkl'), 'rb'))
    model_params['data']['trn_dl_params']['store_coords_in_ram'] = True
    model_params['data']['val_dl_params']['store_coords_in_ram'] = True
    del model_params['gpu_augmentation']['grayvalue']
    model2 = SegmentationModel(val_fold, data_name, 'warm_start_no_gamma',
                               model_parameters=model_params)
    
    # %% opt and lr params
    opt1_params = {'lr': 0.5*10**-4}
    opt2_params = {'momentum': 0.99, 'weight_decay': 3e-5, 'nesterov': True,
                   'lr': 0.5*10**-2}
    lr1_params = {'beta': 0.9, 'lr_min': 10**-6}
    lr2_params = {'beta': 0.9, 'lr_min': 10**-6}
    
    # %%
    model_path = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models',
                              data_name, 'joined_{:.1f}_{}'.format(loss_weight, dose))
    training = JoinedTraining(model1, model2, data.trn_dl,  model_path,
                              loss_weight, num_epochs=500,
                              lr1_params=lr1_params, lr2_params=lr2_params,
                              opt1_params=opt1_params, opt2_params=opt2_params,
                              val_dl=data.val_dl)
    # %% now the magic!!
    training.train()
