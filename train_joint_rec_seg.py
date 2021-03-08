from ovseg.data.JoinedData import JoinedData
import os
from ovseg.training.JoinedTraining import JoinedTraining
from ovseg.model.Reconstruction2dSimModel import Reconstruction2dSimModel
from ovseg.model.SegmentationModel import SegmentationModel
import pickle
import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x
from ovseg.utils import io
from time import sleep

import argparse

# %% get all arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data")
parser.add_argument("--use_windowed_simulations", required=False, default=False,
                    action="store_true")
args = parser.parse_args()

loss_weights = [1.0, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]
model_name_recon = 'recon_fbp_convs'

if args.use_windowed_simulations:
    model_name_recon += '_win'
    proj_folder = 'projections_normal_win'
    im_folder = 'images_HU_win_rescale'
    simulation = 'win'
else:
    proj_folder = 'projections_normal'
    im_folder = 'images_HU_rescale'
    simulation = 'HU'

model_name_recon += '_pretrained'

val_fold = 0
if args.data == 'ov_pod':
    data_name = 'OV04'
    preprocessed_name = 'pod_default'
    model_name_seg = 'segmentation_pretrained'
elif args.data == 'ov_om':
    data_name = 'OV04'
    model_name_seg = 'segmentation_om_pretrained'
    preprocessed_name = 'om_default'
elif args.data.startswith('kits'):
    data_name = 'kits19'
    preprocessed_name = 'default'
    model_name_seg = 'segmentation_pretrained'
else:
    raise ValueError('Found non matching data input {}. Choose one of [ov_pod, ov_om, kits]')

# %% build data
trn_dl_params = {'batch_size': 12, 'patch_size': [512, 512],
                 'num_workers': 12, 'pin_memory': True,
                 'epoch_len': 250, 'store_coords_in_ram': True,
                 'return_fp16': True}
val_dl_params = {'batch_size': 12, 'patch_size': [512, 512],
                 'num_workers': 12, 'pin_memory': True,
                 'epoch_len': 25, 'store_coords_in_ram': True, 'store_data_in_ram': True,
                 'n_max_volumes': 20,
                 'return_fp16': True}
preprocessed_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed',
                                 data_name, preprocessed_name)
keys = ['projection', 'image', 'label', 'spacing']
folders = [proj_folder, im_folder, 'labels', 'orig_spacings']
print('create joint data')
data = JoinedData(val_fold, preprocessed_path, keys, folders,
                  trn_dl_params=trn_dl_params,
                  val_dl_params=val_dl_params)
# %% load models
print('create recon model')
model1 = Reconstruction2dSimModel(val_fold, data_name, model_name_recon,
                                  dont_store_data_in_ram=True)
model_path = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models',
                          data_name, 'segmentation_pretrain')
model_params = pickle.load(open(os.path.join(model_path, 'model_parameters.pkl'), 'rb'))
prep_params = pickle.load(open(os.path.join(preprocessed_path, 'preprocessing_parameters.pkl'),
                               'rb'))
model_params['preprocessing'] = prep_params
print('create segmentation model')
model2 = SegmentationModel(val_fold, data_name, model_name_seg,
                           model_parameters=model_params,
                           dont_store_data_in_ram=True)

# %% opt and lr params
opt1_params = {'lr': 0.5*10**-4, 'betas': (0.9, 0.999)}
opt2_params = {'momentum': 0.99, 'weight_decay': 3e-5, 'nesterov': True,
               'lr': 0.5**0.9 * 0.01}
lr1_params = {'beta': 0.9, 'lr_min': 0}
lr2_params = {'beta': 0.9, 'lr_min': 0}


for loss_weight in loss_weights:
    # %%
    model_path = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models',
                              data_name, 'joined_{}_{}'.format(loss_weight, simulation))
    training = JoinedTraining(model1, model2, data.trn_dl,  model_path,
                              loss_weight, num_epochs=500,
                              lr1_params=lr1_params, lr2_params=lr2_params,
                              opt1_params=opt1_params, opt2_params=opt2_params,
                              val_dl=data.val_dl, fp32=False)
    # %% now the magic!!
    training.train()
    
    # validation. A bit more complicated here. Other models need some improvements...
    results = {}
    model2._init_global_metrics()
    val_path = os.path.join(model_path, 'validation')
    recon_path = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', args.data,
                              'joined_{:.4f}_{}'.format(loss_weight, simulation), 'reconstructions')
    pred_path = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', args.data,
                             'joined_{:.4f}_{}'.format(loss_weight, simulation), 'segmentations')
    for path in [val_path, pred_path, recon_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    val_scans = data.val_scans.copy()
    cases_save_img = val_scans[:10]
    print()
    print()
    for tpl, scan in tqdm(zip(data.val_ds, data.val_scans)):
        with torch.cuda.amp.autocast():
            recon = model1.predict(tpl, return_torch=True)
            recon_prep = model2.preprocessing(recon, tpl['orig_spacing'])
            pred = model2.predict(tpl, model1.pred_key)
        case_id = scan.split('.')[0]
        # compute and store results
        results[case_id] = model2.compute_error_metrics(tpl)
        # model2._update_global_metrics(tpl)
        # maybe save recon and pred
        if scan in cases_save_img:
            io.save_nii(recon.cpu().numpy(),
                        os.path.join(recon_path, case_id),
                        tpl['orig_spacing'])
            io.save_nii(pred,
                        os.path.join(pred_path, case_id+'_pred'),
                        tpl['spacing'])

    model2._save_results_to_pkl_and_txt(results, val_path, ds_name='validation')
