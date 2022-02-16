from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
parser.add_argument("exp", type=int)
args = parser.parse_args()

patch_size = [32, 216, 216]
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

p_name = 'abd_les'
model_names = ['U-Net4', 'U-Net4_new_bin_loss']
loss_names_list = [['cross_entropy', 'dice_loss'],
                   ['cross_entropy', 'dice_loss', 'bin_cross_entropy', 'bin_dice_loss']]

loss_weights_list = [[1.0, 1.0], [0.25, 0.25, 0.75, 0.75]]

loss_names = loss_names_list[args.exp]
loss_weights = loss_weights_list[args.exp]
model_name = model_names[args.exp]

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=8,
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': loss_names,
                                           'loss_weights': loss_weights}

model = SegmentationModel(val_fold=args.vf,
                          data_name='OV04',
                          preprocessed_name=p_name, 
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()
model.eval_raw_data_npz('BARTS')
ens = SegmentationEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name=p_name, 
                           model_name=model_name)

ens.wait_until_all_folds_complete()

ens.eval_raw_dataset('BARTS')
