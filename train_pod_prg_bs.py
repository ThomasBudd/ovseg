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


norm = 'batch' if args.exp == 0 else 'inst'

batch_size_list = [12, 10, 6, 4]

mu = 0.99
p_name = 'pod_067'
model_name = 'prg_bs_'+norm
data_name='OV04'
patch_size = [32, 256, 256]
use_prg_trn = True
out_shape = [[20, 160, 160], [24, 192, 192], [28, 224, 224], [32, 256, 256]]
larger_res_encoder = True
model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                      z_to_xy_ratio=8,
                                                      use_prg_trn=use_prg_trn,
                                                      larger_res_encoder=larger_res_encoder,
                                                      n_fg_classes=1,
                                                      out_shape=out_shape)

model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                           'dice_loss']}

model_params['training']['stop_after_epochs'] = [250, 500, 750]
model_params['training']['opt_params']['momentum'] = mu
model_params['data']['trn_dl_params']['batch_size'] = batch_size_list[0]
model_params['data']['trn_dl_params']['min_biased_samples'] = batch_size_list[0]//2
model_params['data']['val_dl_params']['batch_size'] = batch_size_list[0]
model_params['data']['val_dl_params']['min_biased_samples'] = batch_size_list[0]//2
model_params['network']['norm'] = norm

model = SegmentationModel(val_fold=args.vf,
                          data_name=data_name,
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)



model.training.train()

while model.training.epochs_done < 1000:
    model.training.train()
    for i, ep in enumerate([250, 500, 750]):
        
        if model.training.epochs_done == ep:

            # update the model parameters to the new batch size
            bs = batch_size_list[i+1]
            model.model_parameters['data']['trn_dl_params']['batch_size'] = bs
            model.model_parameters['data']['val_dl_params']['batch_size'] = bs
            model.model_parameters['data']['trn_dl_params']['min_biased_samples'] = bs//2
            model.model_parameters['data']['val_dl_params']['min_biased_samples'] = bs//2
            model.data.trn_dl.dataset.clean()
            if model.data.val_dl is not None:
                model.data.val_dl.dataset.clean()
            # create the new dataloaders
            model.initialise_data()
            # hand over to the training object
            model.training.trn_dl = model.data.trn_dl
            model.training.val_dl = model.data.val_dl
            # update to match the correct folders
            model.training.prg_trn_update_parameters()
            
model.eval_raw_data_npz('BARTS')

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=p_name)

while not ens.all_folds_complete():
    sleep(60)

ens.eval_raw_dataset('BARTS')