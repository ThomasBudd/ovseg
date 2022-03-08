from ovseg.model.SegmentationModel import ClassEnsemblingModel
from ovseg.model.model_parameters_segmentation import get_model_params_class_ensembling
import matplotlib.pyplot as plt
import numpy as np
import torch
debug = False

prev_stages = [{'data_name': 'OV04',
                'preprocessed_name': 'pod_067',
                'model_name': 'larger_res_encoder'},
               {'data_name': 'OV04',
                'preprocessed_name': 'om_08',
                'model_name': 'res_encoder_no_prg_lrn'}]

if debug:
    model_params = get_model_params_class_ensembling(prev_stages, [24, 120, 120], 5, 2)
    model_params['network']['filters'] = 8
    model_params['training']['num_epochs'] = 20
    model_params['data']['trn_dl_params']['epoch_len'] = 25
    model_params['data']['val_dl_params']['epoch_len'] = 2
else:
    model_params = get_model_params_class_ensembling(prev_stages, [32, 160, 160], 5, 2)

model = ClassEnsemblingModel(val_fold=0, data_name='OV04', preprocessed_name='pod_om_10',
                             model_name='class_ensembling_test', model_parameters=model_params)

model.training.train()
model.eval_validation_set()

# %%

# for batch in model.data.trn_dl:
#     break
# batch = batch.cpu().numpy().astype(float)
# z = np.argmax(np.sum(batch[:, 2], (-2,  -1)), 1)
# for i in range(2):
#     for j in range(3):
#         plt.subplot(2, 3, 3*i + j + 1)
#         plt.imshow(batch[i, j, z[i]], cmap='bone')

# # %%

# for batch in model.data.trn_dl:
#     break
# batch = batch.cuda().type(torch.float)
# out = model.network(batch[:, :-1])

# # %%
# plt.figure()
# data_tpl = model.data.trn_ds[0]
# im = np.stack([data_tpl['image'], data_tpl['bin_pred']])
# im = torch.from_numpy(im).cuda()
# pred = model.prediction(im)
# pred_int = np.argmax(pred.cpu().numpy(), 0)
# z = np.argmax(np.sum(pred_int, (1, 2)))
# z = np.argmax(np.sum(data_tpl['bin_pred'], (1, 2)))
# plt.subplot(1, 2, 1)
# plt.imshow(pred_int[z] * data_tpl['bin_pred'][z])
# torch.cuda.empty_cache()
# # %%
# pred2 = model(data_tpl)
# plt.subplot(1, 2, 2)
# plt.imshow(pred2[0,z])
