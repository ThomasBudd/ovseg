from ovseg.model.SLDSExpertModel import SLDSExpertModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.data.Dataset import raw_Dataset
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

p_name = 'SLDS_reg_expert_0.01'

vf_list = [[0], [1, 2], [3, 4]][args.exp]
model_name = 'U-Net2'

for vf in vf_list:
    
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=[8, 48, 48],
                                                         z_to_xy_ratio=5.0/0.67,
                                                         use_prg_trn=False,
                                                         larger_res_encoder=False,
                                                         n_fg_classes=1)
    model_params['network']['filters'] = 128
    model_params['network']['n_blocks_list'] = [6, 3]
        
    model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                              'dice_loss_weighted'],
                                              'loss_kwargs': [{'weight_bg': 1,
                                                               'n_fg_classes': 1},
                                                              {'eps': 1e-5,
                                                               'weight': 1}]}
    
    model_params['data']['folders'] = ['images', 'labels', 'regions']
    model_params['data']['keys'] = ['image', 'label', 'region']
    model_params['training']['batches_have_masks'] = True
    model_params['training']['num_epochs'] = 1000
    model_params['training']['stop_after_epochs'] = list(range(100, 1000, 100))
    model_params['postprocessing'] = {'mask_with_reg': True}
    
    
    model = SLDSExpertModel(val_fold=vf,
                            data_name='OV04',
                            preprocessed_name=p_name, 
                            model_name=model_name,
                            model_parameters=model_params)
    
    while model.training.epochs_done < 900:
        model.training.train()
        BARTS_ds = raw_Dataset(os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'BARTS'),
                               prev_stages=model.prev_stages)
        model.eval_ds(model.data.val_ds, 'validation_{}'.format(model.training.epochs_done),
                      save_preds=False)
        model.eval_ds(BARTS_ds, 'BARTS_{}'.format(model.training.epochs_done),
                      save_preds=False)
    model.training.train()
    model.eval_validation_set()
    model.eval_raw_dataset('BARTS')
