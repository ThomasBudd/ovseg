from ovseg.model.SLDSModel import SLDSModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
parser.add_argument("exp", type=int)
args = parser.parse_args()

w_list = [[0.001], [0.01, 0.1]][args.exp]
patch_size = [40, 320, 320]
use_prg_trn = True
out_shape = [[24, 192, 192],
             [28, 224, 224],
             [36, 288, 288],
             [40, 320, 320]]
larger_res_encoder = True

for w in w_list:
    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                         z_to_xy_ratio=5.0/0.67,
                                                         use_prg_trn=use_prg_trn,
                                                         larger_res_encoder=larger_res_encoder,
                                                         n_fg_classes=2,
                                                         out_shape=out_shape)
    model_params['training']['loss_params'] = {'loss_names': ['SLDS_loss'],
                                              'loss_kwargs': [{'weight_bg': w,
                                                               'n_fg_classes': 2}]}
    model_params['data']['folders'] = ['images', 'labels', 'regions']
    model_params['data']['keys'] = ['image', 'label', 'region']
    # we train using the regions as ground truht we're training for
    model_params['data']['trn_dl_params']['label_key'] = 'region'
    model_params['data']['val_dl_params']['label_key'] = 'region'
    
    
    model = SLDSModel(val_fold=args.vf,
                      data_name='OV04',
                      preprocessed_name='SLDS',
                      model_name='U-Net5_'+str(w),
                      model_parameters=model_params)
    
    model.training.train()
    model.eval_validation_set(save_preds=True, save_plots=False)    
    model.eval_raw_data_npz('BARTS')
