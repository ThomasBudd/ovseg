from ovseg.model.ContourRefinementV3Model import ContourRefinementV3Model
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

patch_size = [24, 96, 96]

sizes = 16*np.round(patch_size[2] / np.arange(4,0,-1)**(1/3) / 16)
out_shape = [ [int(s)//4,  int(s), int(s)] for s in sizes]
print(out_shape)

data_name = 'kits21'
p_name = 'kidney_full_refine_new'

bs_list = [8, 12, 16]
wd_list = [1e-4, 1e-4, 2e-4]


for bs, wd in zip(bs_list, wd_list):
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                         z_to_xy_ratio=3.0/0.8,
                                                         use_prg_trn=True,
                                                         out_shape=out_shape,
                                                         larger_res_encoder=False,
                                                         n_fg_classes=1)
    
    model_params['architecture'] = 'UNet'
    model_params['network']['kernel_sizes'] = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    model_params['network']['in_channels'] = 2
    model_params['network']['norm'] = 'batch'
    del model_params['network']['block']
    del model_params['network']['z_to_xy_ratio']
    del model_params['network']['n_blocks_list']
    del model_params['network']['stochdepth_rate']
    
    model_params['data']['folders'] = ['images', 'labels', 'regions', 'prev_preds']
    model_params['data']['keys'] = ['image', 'label', 'region', 'prev_pred']
    
    for s in ['trn_dl_params', 'val_dl_params']:
        model_params['data'][s]['batch_size'] = bs
        model_params['data'][s]['min_biased_samples'] = bs//3
        model_params['data'][s]['num_workers'] = 14
    model_params['training']['batches_have_masks'] = True
    model_params['training']['opt_params']['weight_decay'] = wd
    model_params['training']['opt_params']['momentum'] = 0.9
    
    model_params['postprocessing'] = {'mask_with_reg': True}
    
    model_name = f'refine_model_bs_{bs}'
    
    model = ContourRefinementV3Model(val_fold=args.vf,
                                      data_name=data_name,
                                      preprocessed_name=p_name, 
                                      model_name=model_name,
                                      model_parameters=model_params)
    
    
    model.training.train()
    model.eval_validation_set()
