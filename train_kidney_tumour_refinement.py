from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import numpy as np
import matplotlib.pyplot as plt
import argparse
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

patch_size = [24, 96, 96]

sizes = 16*np.round(patch_size[2] / np.arange(4,0,-1)**(1/3) / 16)
out_shape = [ [int(s)//4,  int(s), int(s)] for s in sizes]
print(out_shape)

data_name = 'kits21'
p_name = 'kidney_tumour_refine'

bs = 8
wd = 1e-4

combinations = list(product(range(5), [0, 0.5, 1.0]))

vf_p_list = combinations[args.i::8]

for vf, p in vf_p_list:
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                         z_to_xy_ratio=3.0/0.8,
                                                         use_prg_trn=False,
                                                         larger_res_encoder=False,
                                                         n_fg_classes=1)
    model_params['network']['in_channels'] = 2
    model_params['network']['norm'] = 'batch'
    model_params['network']['n_blocks_list'] = [2, 6, 3]
    
    model_params['data']['folders'] = ['images', 'labels', 'masks', 'prev_preds']
    model_params['data']['keys'] = ['image', 'label', 'mask', 'prev_pred']
    
    for s in ['trn_dl_params', 'val_dl_params']:
        model_params['data'][s]['batch_size'] = bs
        model_params['data'][s]['min_biased_samples'] = bs//3
        model_params['data'][s]['num_workers'] = 8
        model_params['data'][s]['bias'] = 'error'
        model_params['data'][s]['p_weighted_volume_sampling'] = p
        del model_params['data'][s]['store_coords_in_ram']
        del model_params['data'][s]['memmap']
    model_params['training']['batches_have_masks'] = True
    model_params['training']['opt_params']['weight_decay'] = wd
    model_params['training']['opt_params']['momentum'] = 0.9
    
    model_params['postprocessing'] = {'mask_with_reg': True}
    
    model_name = f'refine_p_{p}'
    
    model = SegmentationModelV2(val_fold=vf,
                                data_name=data_name,
                                preprocessed_name=p_name, 
                                model_name=model_name,
                                model_parameters=model_params)
    
    
    model.training.train()
    model.eval_validation_set()
