from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("bs", type=int)
parser.add_argument("--small", type=bool, default=False, action='store_true')
args = parser.parse_args()

p_name='pod_067'
bs = args.bs

if args.small:
    patch_size = [32, 216, 216] 
    larger_res_encoder = False
else:
    patch_size = [32, 256, 256]
    larger_res_encoder = True
use_prg_trn = False
out_shape = None#[[20, 160, 160], [24, 192, 192], [28, 224, 224], [32, 256, 256]]


model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                      z_to_xy_ratio=5.0/0.67,
                                                      use_prg_trn=use_prg_trn,
                                                      larger_res_encoder=larger_res_encoder,
                                                      n_fg_classes=1,
                                                      out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                           'dice_loss']}

model_params['architecture'] = 'unetresstemencoder'
model_params['data']['trn_dl_params']['min_biased_samples'] = int(bs/3+0.5)
model_params['data']['val_dl_params']['min_biased_samples'] = int(bs/3+0.5)
model_params['data']['trn_dl_params']['batch_size'] = bs
model_params['data']['val_dl_params']['batch_size'] = bs
model_name = 'test_bs_' + str(bs)

model = SegmentationModel(val_fold=args.vf,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)

model.training.train()
