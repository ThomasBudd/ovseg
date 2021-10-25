from ovseg.model.ClassCascadeModel import ClassCascadeModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
parser.add_argument("i", type=int)
args = parser.parse_args()

pref_list = 3 * ['pod_om'] + ['pod']

lb_classes_list = [[2, 13, 15, 17], [2], [1, 2, 9], [1]]
pref = pref_list[args.i]
lb_classes = lb_classes_list[args.i]

data_name = 'OV04'
p_name='cascade_'+pref+'_'+'_'.join([str(c) for c in lb_classes])
model_name='U-Net5'
patch_size = [40, 320, 320]
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
                                                     n_fg_classes=len(lb_classes),
                                                     out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                          'dice_loss']}
model_params['network']['in_channels'] = 2
model_params['data']['folders'] = ['images', 'labels', 'prev_preds']
model_params['data']['keys'] = ['image', 'label', 'prev_pred']
model_params['training']['batches_have_masks'] = True
model_params['postprocessing'] = {'mask_with_reg': True}
model_params['data']['val_dl_params']['bias'] = 'cl_fg'
model_params['data']['trn_dl_params']['bias'] = 'cl_fg'
model_params['data']['val_dl_params']['n_fg_classes'] = len(lb_classes)
model_params['data']['trn_dl_params']['n_fg_classes'] = len(lb_classes)

model = ClassCascadeModel(val_fold=args.vf, 
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=p_name,
                          model_paramters=model_params)
model.training.train()
model.eval_validation_set()
model.eval_raw_data_npz('BARTS')


