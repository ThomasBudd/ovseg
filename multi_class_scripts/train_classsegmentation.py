from ovseg.model.ClassSegmentationModel import ClassSegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

patch_size = [40, 320, 320]
model_name = 'U-Net5'
use_prg_trn = True
out_shape = [[24, 192, 192],
             [28, 224, 224],
             [36, 288, 288],
             [40, 320, 320]]
larger_res_encoder = True
model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                      z_to_xy_ratio=5.0/0.67,
                                                      use_prg_trn=use_prg_trn,
                                                      larger_res_encoder=larger_res_encoder,
                                                      n_fg_classes=6,
                                                      out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                           'dice_loss']}
model_params['network']['in_channels'] = 2
model_params['data']['folders'] = ['images', 'labels', 'prev_preds']
model_params['data']['keys'] = ['image', 'label', 'prev_pred']
model_params['training']['batches_have_masks'] = True
model_params['postprocessing'] = {'mask_with_reg': True}
model_params['data']['val_dl_params']['n_fg_classes'] = 6
model_params['data']['trn_dl_params']['n_fg_classes'] = 6

model = ClassSegmentationModel(val_fold=args.vf,
                               data_name='OV04',
                               preprocessed_name='ClassSegmentation',
                               model_name='U-Net5',
                               model_parameters=model_params)

model.training.train()
model.eval_raw_data_npz('BARTS')