from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.model.RegionfindingEnsemble import RegionfindingEnsemble
from ovseg.preprocessing.RegionexpertPreprocessing import RegionexpertPreprocessing
from time import sleep

import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("vf", type=int)
# args = parser.parse_args()

w = 0
vf = 0#args.vf

p_name = 'multiclass_reg'
patch_size = [40, 320, 320]
use_prg_trn = False
out_shape = None
larger_res_encoder = True
model_params = get_model_params_3d_res_encoder_U_Net(patch_size, 
                                                     5/0.67,
                                                     n_fg_classes=6,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     out_shape=out_shape)

model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                          'dice_loss_weighted'],
                                          'loss_kwargs': [{'weight_bg': 0,
                                                           'n_fg_classes': 6},
                                                          {'eps': 1e-5,
                                                           'weight': 0}]}
model_params['data']['folders'] = ['images', 'labels', 'regions']
model_params['data']['keys'] = ['image', 'label', 'region']
# we train using the regions as ground truht we're training for
for dl_str in ['trn_dl_params', 'val_dl_params']:
    model_params['data'][dl_str]['label_key'] = 'region'
    model_params['data'][dl_str]['bias'] = 'cl_fg'
    model_params['data'][dl_str]['n_fg_classes'] = 6
    model_params['data'][dl_str]['min_biased_samples'] = 2
model_params['network']['filters'] = 4
model_params['network']['use_logit_bias'] = True

model = RegionfindingModel(val_fold=vf,
                           data_name='OV04_test',
                           preprocessed_name=p_name,
                           model_name='regfinding_'+str(w),
                           model_parameters=model_params)

# %%
import numpy as np
from tqdm import tqdm
n_cl = np.zeros(6)
# %%
for batch in tqdm(model.data.val_dl):
    lb = batch[:, -1].cpu().numpy()
    for cl in range(1,7):
        n_cl[cl-1] += np.sum(np.max(lb == cl, (1,2,3)))

print(n_cl)

# %%
import matplotlib.pyplot as plt

for batch in model.data.val_dl:
    batch = batch[0].cpu().numpy().astype(float)
    break

plt.imshow(batch[0, 20, 160:-160, 160:-160],cmap='bone')
plt.contour(batch[1, 20, 160:-160, 160:-160])
# %%
import torch
for batch in model.data.val_dl:
    batch = batch[:, :1, 10:-10, 160:-160, 160:-160].cuda().type(torch.float)
    break

with torch.no_grad():
    logsl1 = model.network(batch)
for log_layer in model.network.all_logits:
    w = log_layer.logits.weight.clone()
    w[0] = 0
    log_layer.logits.weight = torch.nn.Parameter(w)
    b = log_layer.logits.bias.clone()
    b[0] = -50
    log_layer.logits.bias = torch.nn.Parameter(b)
with torch.no_grad():
    logsl2 = model.network(batch)

sm = torch.nn.functional.softmax(logsl2[-1], 1)
