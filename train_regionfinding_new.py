from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.model.RegionfindingEnsemble import RegionfindingEnsemble

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("w", type=int)
args = parser.parse_args()

# w_list = 4*[0.01] + 2*[1e-3, 1e-4, 1e-5]
w_list = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.01]
w = w_list[args.w]
# vf_list = list(range(5)) + 4 * [0]
vf = 0 #vf_list[args.w]

model_params = get_model_params_3d_res_encoder_U_Net([32, 216, 216], 5/0.8, n_fg_classes=11,
                                                     larger_res_encoder=False)

model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                          'dice_loss_weighted'],
                                          'loss_kwargs': [{'weight_bg': w,
                                                           'n_fg_classes': 11},
                                                          {'eps': 1e-5,
                                                           'weight': w}]}
model_params['data']['folders'] = ['images', 'labels', 'regions']
model_params['data']['keys'] = ['image', 'label', 'region']
# we train using the regions as ground truht we're training for
model_params['data']['trn_dl_params']['label_key'] = 'region'
model_params['data']['val_dl_params']['label_key'] = 'region'
# model_params['training']['num_epochs'] = 10
# model_params['network']['filters'] = 8
# model_params['data']['trn_dl_params']['epoch_len'] = 50
# model_params['data']['val_dl_params']['epoch_len'] = 5


model = RegionfindingModel(val_fold=vf, data_name='OV04_test',
                           preprocessed_name='multiclass_reg',
                           model_name='regfinding_'+str(w), model_parameters=model_params)

model.training.train()
model.eval_validation_set(save_preds=True, save_plots=False)
model.eval_raw_data('BARTS', save_preds=True)

# # %%
# import matplotlib.pyplot as plt
# for batch in model.data.val_dl:
#     break

# batch = batch.numpy().astype(float)

# plt.imshow(batch[0, 0, 16], cmap='gray')
# plt.contour(batch[0, 1, 16], colors='red',linewidths=0.5)

# # %%
# import numpy as np
# data_tpl = model.data.val_ds[0]
# im = data_tpl['image'][0].astype(float)
# lb = (data_tpl['label'][0] > 0).astype(float)
# reg = (data_tpl['region'][0] > 0).astype(float)
# z = np.argmax(np.sum(lb, (1, 2)))
# plt.imshow(im[z], cmap='gray')
# plt.contour(lb[z], colors='red',linewidths=0.5)
# plt.contour(reg[z], colors='red',linewidths=0.25, linestyles='dashed')