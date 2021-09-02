from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int)
parser.add_argument("p", type=int)
parser.add_argument("--no_cascade", required=False, default=False, action='store_true')
args = parser.parse_args()

p_name = ['pod_om_cascade_08', 'multiclass_cascade_08'][args.p]
n_fg_classes = [2, 11][args.p]

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=[32, 216, 216],
                                                     z_to_xy_ratio=5.0/0.8,
                                                     use_prg_trn=False,
                                                     n_fg_classes=n_fg_classes)
if args.no_cascade:
    model_name='res_encoder_no_cascade'
else:
    model_name='res_encoder'
    model_params['data']['folders'] = ['images', 'labels', 'bin_preds']
    model_params['data']['keys'] = ['image', 'label', 'bin_pred']
    model_params['data']['trn_dl_params']['pred_fps_key'] = 'bin_pred'
    # we set this to have only one foreground class here as the input is binary
    model_params['data']['trn_dl_params']['n_pred_classes'] = 1
    model_params['data']['val_dl_params']['pred_fps_key'] = 'bin_pred'
    model_params['data']['val_dl_params']['n_pred_classes'] = 1
    model_params['network']['in_channels'] = 2
    


model = SegmentationModel(val_fold=args.gpu,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)
if args.no_cascade:
    if 'prev_stages' in model.model_parameters['preprocessing']:
        del model.model_parameters['preprocessing']['prev_stages']
    if 'prev_stages' in model.model_parameters:
        del model.model_parameters['prev_stages']
    if hasattr(model, 'prev_stages'):
        del model.prev_stages
    model.preprocessing.prev_stages = []
    model.save_model_parameters()
model.training.train()
model.eval_validation_set()
model.eval_raw_data_npz('BARTS')
model.clean()
