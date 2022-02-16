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

data_name = 'OV04'
p_name = 'pod_om_08_5'

model_name = 'test_ufl_'+str(args.exp)

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5/0.8,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=2,
                                                     out_shape=out_shape)

if args.exp == 0:
    model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                              'modifiedTverskyLoss']}
elif args.exp == 1:
    
    model_params['training']['loss_params'] = {'loss_names': ['modifiedFocalLoss',
                                                              'modifiedTverskyLoss']}
elif args.exp == 2:
    
    model_params['training']['loss_params'] = {'loss_names': ['modifiedFocalLoss',
                                                              'modifiedTverskyLoss'],
                                               'loss_kwargs': [{'scale':1.0}, {}]}
model_params['data']['val_dl_params']['n_fg_classes'] = 2
model_params['data']['trn_dl_params']['n_fg_classes'] = 2
model_params['data']['val_dl_params']['bias'] = 'cl_fg'
model_params['data']['trn_dl_params']['bias'] = 'cl_fg'
model_params['architecture'] = 'unetresencoder'

model = SegmentationModel(val_fold=args.vf,
                          data_name=data_name,
                          preprocessed_name=p_name, 
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_raw_data_npz('BARTS')
ens = SegmentationEnsemble(val_fold=list(range(5,8)),
                           data_name=data_name,
                           preprocessed_name=p_name, 
                           model_name=model_name)

ens.wait_until_all_folds_complete()

ens.eval_raw_dataset('BARTS')
