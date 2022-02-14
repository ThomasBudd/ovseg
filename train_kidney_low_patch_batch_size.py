from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i", type=int)
parser.add_argument("vf", type=int)
args = parser.parse_args()

data_name = 'kits21'
p_name = 'kidney_full'

if args.i == 0:
    patch_size_list = [[128, 512, 512], [96, 384, 384]]
    batch_size_list = [2, 4]
    wd_list = [3e-5, 1e-4]
    min_biased_samples_list = [1, 1]
    momentum_list = [0.99, 0.98]
    norm = 'inst'
else:
    patch_size_list = [[64, 256, 256], [64, 256, 256]]
    batch_size_list = [12, 16]
    wd_list = [2e-4, 2e-4]
    min_biased_samples_list = [4, 5]
    momentum_list = [0.9, 0.9]
    norm = 'batch'


for patch_size, batch_size, wd, min_biased_samples, momentum in zip(patch_size_list,
                                                                    batch_size_list,
                                                                    wd_list,
                                                                    min_biased_samples_list,
                                                                    momentum_list):
    
    sizes = 64*np.floor(patch_size[2] / np.arange(4,0,-1)**(1/3) / 64)
    out_shape = [ [int(s)//4,int(s),int(s)] for s in sizes]
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                         z_to_xy_ratio=3.0/0.8,
                                                         use_prg_trn=True,
                                                         out_shape=out_shape,
                                                         larger_res_encoder=False,
                                                         n_fg_classes=1)
    
    model_params['architecture'] = 'UNet'
    model_params['network']['kernel_sizes'] = 5*[(3, 3, 3)]
    model_params['network']['norm'] = norm
    model_params['network']['stem_kernel_size'] = [1, 4, 4]
    del model_params['network']['block']
    del model_params['network']['z_to_xy_ratio']
    del model_params['network']['n_blocks_list']
    del model_params['network']['stochdepth_rate']
    
    model_params['data']['folders'] = ['images', 'labels']
    model_params['data']['keys'] = ['image', 'label']
    model_params['data']['trn_dl_params']['batch_size'] = batch_size
    model_params['data']['trn_dl_params']['min_biased_samples'] = 4
    model_params['training']['opt_params']['weight_decay'] = wd
    model_params['training']['opt_params']['momentum'] = momentum
    model_params['training']['save_additional_weights_after_epochs'] = [750]
    
    
    model_name = f'ps_{patch_size[0]}_bs{batch_size}'

    model = SegmentationModel(val_fold=args.vf,
                              data_name=data_name,
                              preprocessed_name=p_name, 
                              model_name=model_name,
                              model_parameters=model_params)
    
    
    model.training.train()
    model.eval_validation_set()
