from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

patch_size = [40, 160, 160]
model_name = 'U-Net5'
use_prg_trn = True
out_shape = [[24, 96, 96],
             [32, 128, 128],
             [40, 160, 160]]
larger_res_encoder = True

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/1.6,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=1,
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                          'dice_loss']}

model_params['augmentation']['torch_params']['grid_inplane']['p_rot'] = 1.0
model_params['data']['trn_dl_params']['store_data_in_ram'] = True

p_name = 'default'
model = SegmentationModel(val_fold=args.vf,
                          data_name='Lits_5mm',
                          preprocessed_name=p_name, 
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_raw_dataset('OV04', save_preds=True, save_plots=True)
model.eval_raw_dataset('BARTS', save_preds=True, save_plots=True)