from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
parser.add_argument("exp", type=int)
args = parser.parse_args()

patch_size = [40, 320, 320]
model_name = 'U-Net5'
use_prg_trn = True
out_shape = [[24, 192, 192],
             [28, 224, 224],
             [36, 288, 288],
             [40, 320, 320]]
larger_res_encoder = True

lb_classes = [1, 2, 9, 13, 15, 17]

p_name = 'multiclass_1_2_9_13_15_17'

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/0.67,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=len(lb_classes),
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                          'dice_loss']}

model_params['data']['val_dl_params']['n_fg_classes'] = len(lb_classes)
model_params['data']['trn_dl_params']['n_fg_classes'] = len(lb_classes)
model_params['data']['val_dl_params']['bias'] = 'cl_fg'
model_params['data']['trn_dl_params']['bias'] = 'cl_fg'

if args.exp == 0:
    model_name += '_long'
    model_params['training']['num_epochs'] = 2000
else:
    model_name += '_bs_4'
    model_params['data']['trn_dl_params']['batch_size'] = 4
    model_params['data']['trn_dl_params']['min_biased_samples'] = 2
    model_params['data']['val_dl_params']['batch_size'] = 4
    model_params['data']['val_dl_params']['min_biased_samples'] = 2

p_name = 'multiclass_'+'_'.join([str(c) for c in lb_classes])
model = SegmentationModel(val_fold=args.vf,
                          data_name='OV04',
                          preprocessed_name=p_name, 
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()


ens = SegmentationEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name=p_name, 
                           model_name=model_name)

while not ens.all_folds_complete():
    sleep(60)

ens.eval_raw_dataset('BARTS')


