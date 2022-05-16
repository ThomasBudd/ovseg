from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
# parser.add_argument("exp", type=int)
args = parser.parse_args()

vf = args.vf
# if args.exp == 0:
#     w_list_list = [[-2, -2], [0, 0]]
# else:
#     w_list_list = [[-1, -1], [1, 1]]

data_name = 'OV04'
preprocessed_name = 'pod_om_4fCV'

# equally scale down the patches to have ~half of the pixeld
patch_size = [32, 216, 216]
use_prg_trn = True

out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     out_shape=out_shape,
                                                     n_fg_classes=2,
                                                     use_prg_trn=use_prg_trn)

model_params['data']['n_folds'] = 4
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['training']['opt_params']['momentum'] = 0.98
model_params['training']['opt_params']['weight_decay'] = 1e-4
model_params['training']['loss_params']['loss_names'] = ['dice_loss_sigm_weighted',
                                                         'cross_entropy_exp_weight']

    
model_params['training']['loss_params']['loss_kwargs'] = 2*[{'w_list':[-1.284, -0.544]}]
# change the model name when using other hyper-paramters
model_name = 'new_loss_w_DSC'

model = SegmentationModel(val_fold=vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()
model.eval_raw_data_npz('BARTS')
model.eval_raw_data_npz('ApolloTCGA')

if vf < 4:
    ens = SegmentationEnsemble(val_fold=list(range(4)),
                               data_name=data_name,
                               model_name=model_name,
                               preprocessed_name=preprocessed_name)
else:
    ens = SegmentationEnsemble(val_fold=[4,5,6],
                               data_name=data_name,
                               model_name=model_name,
                               preprocessed_name=preprocessed_name)
    
ens.wait_until_all_folds_complete()
ens.eval_raw_dataset('BARTS')
ens.eval_raw_dataset('ApolloTCGA')
