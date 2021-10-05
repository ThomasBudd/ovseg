from ovseg.model.SLDSExpertModel import SLDSExpertModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
parser.add_argument("vf", type=int)
args = parser.parse_args()

w_list = [[0.001, 0.01], [0.1]][args.exp]
vf = args.vf


for w in w_list:
    
    p_name = 'SLDS_reg_expert_{}'.format(w)
    model_name = 'U-Net2'
    
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
    # model_params['training']['stop_after_epochs'] = list(range(100, 1000, 100))
    model_params['postprocessing'] = {'mask_with_reg': True}
    
    
    model = SLDSExpertModel(val_fold=vf,
                            data_name='OV04',
                            preprocessed_name=p_name, 
                            model_name=model_name,
                            model_parameters=model_params)
    
    model.training.train()
    model.eval_validation_set()
    model.eval_raw_dataset('BARTS')
