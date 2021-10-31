from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
parser.add_argument("exp", type=int)
args = parser.parse_args()

patch_size = [64, 216, 216]
model_name = 'U-Net4_prg_lrn'
use_prg_trn = True
out_shape = [[40, 128, 128],
             [44, 152, 152],
             [60, 192, 192],
             [64, 216, 216]]
larger_res_encoder = False

if args.exp == 0:
    data_name = 'BARTS_dcm_OV04_dcm'
else:
    data_name = 'ApolloTCGA_dcm_BARTS_dcm_OV04_dcm'

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=2.5/0.8,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=2,
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                          'dice_loss']}
model_params['data']['val_dl_params']['n_fg_classes'] = 2
model_params['data']['trn_dl_params']['n_fg_classes'] = 2
model_params['data']['val_dl_params']['bias'] = 'cl_fg'
model_params['data']['trn_dl_params']['bias'] = 'cl_fg'

p_name = 'pod_om_08_25'
model = SegmentationModel(val_fold=args.vf,
                          data_name=data_name,
                          preprocessed_name=p_name, 
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()

if args.exp == 0:
    model.eval_raw_data_npz('ApolloTCGA_dcm')

    ens = SegmentationEnsemble(val_fold=list(range(5, 10)),
                               data_name='OV04',
                               preprocessed_name=p_name, 
                               model_name=model_name)
    
    while not ens.all_folds_complete():
        sleep(60)
    
    ens.eval_raw_dataset('ApolloTCGA_dcm')