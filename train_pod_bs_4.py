from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.utils.io import load_pkl, save_pkl
import argparse
from time import sleep
import os
parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
parser.add_argument("exp", type=int)
args = parser.parse_args()

mu = [0.99, 0.98, 0.97][args.exp]

p_name = 'pod_067'
model_name = 'bs_4_mu_'+str(mu)
data_name='OV04'
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
                                                      n_fg_classes=1,
                                                      out_shape=out_shape)

model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                           'dice_loss']}

model_params['training']['opt_params']['momentum'] = mu
model_params['data']['trn_dl_params']['batch_size'] = 4
model_params['data']['trn_dl_params']['min_biased_samples'] = 2
model_params['data']['val_dl_params']['batch_size'] = 4
model_params['data']['val_dl_params']['min_biased_samples'] = 2

model = SegmentationModel(val_fold=args.vf,
                          data_name=data_name,
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_raw_data_npz('BARTS')

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=p_name)

while not ens.all_folds_complete():
    sleep(60)

ens.eval_raw_dataset('BARTS')