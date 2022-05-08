from ovseg.model.ContourRefinementV3Model import ContourRefinementV3Model
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

patch_sizes = [[16, 64, 64],[24, 96, 96],[32, 128, 128]]

data_name = 'kits21'
p_name = 'kidney_full_refine_v3'
for patch_size in patch_sizes:
    
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                         z_to_xy_ratio=3.0/0.8,
                                                         use_prg_trn=False,
                                                         larger_res_encoder=False,
                                                         n_fg_classes=1)
    
    model_params['architecture'] = 'UNet'
    model_params['network']['kernel_sizes'] = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    model_params['network']['in_channels'] = 2
    
    del model_params['network']['block']
    del model_params['network']['z_to_xy_ratio']
    del model_params['network']['n_blocks_list']
    del model_params['network']['stochdepth_rate']
    
    model_params['data']['folders'] = ['images', 'labels', 'regions', 'prev_preds']
    model_params['data']['keys'] = ['image', 'label', 'region', 'prev_pred']
    model_params['training']['batches_have_masks'] = True
    model_params['training']['stop_after_epochs'] = [200, 400, 600, 800]
    model_params['postprocessing'] = {'mask_with_reg': True}
    
    model_name = f'UNet_{patch_size[2]}'
    
    model = ContourRefinementV3Model(val_fold=args.vf,
                                      data_name=data_name,
                                      preprocessed_name=p_name, 
                                      model_name=model_name,
                                      model_parameters=model_params)
    
    while model.training.epochs_done < 1000:
    
        model.training.train()
        model.eval_ds(model.data.val_ds,
                      f'validation_{model.training.epochs_done}',
                      save_preds=False,
                      merge_to_CV_results=True)