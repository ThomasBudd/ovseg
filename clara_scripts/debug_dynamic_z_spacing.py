import torch
import torch.nn.functional as F
import numpy as np
import os
from ovseg.utils.io import load_pkl, read_dcms
from ovseg.model.ClaraWrappers import preprocess_dynamic_z_spacing
import matplotlib.pyplot as plt

# %%
path_to_dcms1 = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\ICON8_14_Derby_Burton\\ICON8_14_52051381_1'
data_tpl1 = read_dcms(path_to_dcms1)

path_to_dcms2 = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\ICON8_14_Derby_Burton\\ICON8_14_26850164_1'
data_tpl2 = read_dcms(path_to_dcms2)

# %%
path_to_model_params = 'D:\\PhD\\Data\\ov_data_base\\trained_models\\OV04\\pod_om_08_5\\bs4\\model_parameters.pkl'
model_params = load_pkl(path_to_model_params)
model_params['preprocessing']['apply_pooling'] = True
model_params['preprocessing']['pooling_stride'] = [1,2,2]
window = model_params['preprocessing']['window']
# %%

im_list1 = preprocess_dynamic_z_spacing(data_tpl1,
                                        model_params['preprocessing'])
im_list1 = [im[0].cpu().numpy() for im in im_list1]


im_list2 = preprocess_dynamic_z_spacing(data_tpl2,
                                        model_params['preprocessing'])
im_list2 = [im[0].cpu().numpy() for im in im_list2]

# %%
plt.subplot(1,2,1)
plt.imshow(data_tpl2['image'][0].clip(*window), cmap='gray')
plt.subplot(1,2,2)
plt.imshow(im_list2[0][0], cmap='gray')
# %%
plt.subplot(1,3,1)
plt.imshow(data_tpl1['image'][0].clip(*window), cmap='gray')
plt.subplot(1,3,2)
plt.imshow(im_list1[0][0], cmap='gray')
plt.subplot(1,3,3)
plt.imshow(im_list1[1][0], cmap='gray')

# %%
# from time import perf_counter

# pred = (torch.rand((85, 512, 512)).cuda() > 0.99).type(torch.float)

# pred2 = torch.zeros_like(pred)

# st = perf_counter()
# pred2 = pred2 + 9*(pred == 1).type(torch.float)
# print(perf_counter()-st)

# pred2 = torch.zeros_like(pred)

# st = perf_counter()
# pred2[pred == 1] = 9
# print(perf_counter()-st)

# %%
# pred_np = np.random.randn(85, 512, 512)

# st = perf_counter()
# np.moveaxis(pred_np, 0, -1)
# print(perf_counter() - st)

# pred_t = torch.randn(85, 512, 512)

# st = perf_counter()
# torch.stack([pred_t[z] for z in range(pred_t.shape[0])], -1)
# print(perf_counter() - st)