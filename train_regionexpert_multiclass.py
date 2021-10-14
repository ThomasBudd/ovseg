from ovseg.model.RegionexpertModel import RegionexpertModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.model.RegionexpertEnsemble import RegionexpertEnsemble
import sleep

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

w = 0
vf = args.vf

p_name = 'multiclass_reg_expert'
patch_size = [40, 320, 320]
use_prg_trn = True
out_shape = [[24, 192, 192],
             [28, 224, 224],
             [36, 288, 288],
             [40, 320, 320]]
larger_res_encoder = True
model_params = get_model_params_3d_res_encoder_U_Net(patch_size, 
                                                     5/0.67,
                                                     n_fg_classes=1,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                          'dice_loss']}
model_params['data']['folders'] = ['images', 'labels', 'regions']
model_params['data']['keys'] = ['image', 'label', 'region']
model_params['training']['batches_have_masks'] = True
model_params['training']['num_epochs'] = 1000
model_params['postprocessing'] = {'mask_with_reg': True}
model = RegionexpertModel(val_fold=args.vf,
                          data_name='OV04',
                          preprocessed_name=p_name, 
                          model_name='U-Net5',
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()
model.eval_raw_data_npz('BARTS')

if vf == 0:
    ens = RegionexpertEnsemble(val_fold=list(range(5)),
                               data_name='OV04',
                               model_name='U-Net5',
                               preprocessed_name=p_name)

    while not ens.all_folds_complete():
        print('Wait 1 min...')
        sleep(60)
    
    ens.eval_raw_dataset('BARTS')