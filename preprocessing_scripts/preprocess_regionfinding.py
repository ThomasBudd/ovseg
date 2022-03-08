from ovseg.preprocessing.RegionfindingPreprocessing import RegionfindingPreprocessing

prep = RegionfindingPreprocessing(apply_resizing=True,
                                  apply_pooling=False,
                                  apply_windowing=True,
                                  mask_dist=[2, 15, 15],
                                  lb_classes=[1, 9, 2, 3, 4, 5, 6, 7, 13, 14, 15])
prep.plan_preprocessing_raw_data('OV04')
prep.preprocess_raw_data('OV04', 'multiclass')


# %%
# import os
# import matplotlib.pyplot as plt
# import numpy as np

# dp = 'D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04_test\\multiclass'
# case = 'case_004.npy'

# im = np.load(os.path.join(dp, 'images', case)).astype(float)
# lb = np.load(os.path.join(dp, 'labels', case)).astype(float)
# mask = np.load(os.path.join(dp, 'masks', case)).astype(float)

# im = (im - im.min()) / (im.max() - im.min())

# z = np.argmax(np.sum(lb > 0, (1,2)))
# contains = np.where(np.sum(mask == 0, (1,2)))[0]
# # %%
# z = np.random.choice(contains)
# plt.imshow(im[z], cmap='bone')
# plt.contour(lb[z] > 0, colors='r')
# plt.imshow(np.stack([0.7*im[z], 0.7*im[z],0.7*im[z] + 0.3*(1-mask[z])],-1))
# %%

# from ovseg.utils.io import read_data_tpl_from_nii

# data_tpl = read_data_tpl_from_nii(os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
#                                                 'OV04'), 'case_000')
# from time import perf_counter

# #%%
# t1 = perf_counter()
# mask = prep.get_mask_from_data_tpl(data_tpl)
# t2 = perf_counter() - t1
# print(t2)

# # %%
# t1 = perf_counter()
# vol = prep(data_tpl)
# t2 = perf_counter() - t1
# print(t2)
# # %%
# from scipy.ndimage.morphology import binary_dilation
# t1 = perf_counter()
# mask1 = binary_dilation(data_tpl['label'] > 0, prep.selem).astype(float)
# t2 = perf_counter() - t1
# print(t2)
# # %%
# from skimage.morphology import dilation
# t1 = perf_counter()
# mask2 = dilation(data_tpl['label'] > 0, prep.selem).astype(float)
# t2 = perf_counter() - t1
# print(t2)
# # %%
# import torch
# lb_cuda = torch.from_numpy(data_tpl['label']).type(torch.float).cuda()
# bin_lb_cuda = (lb_cuda > 0).type(torch.float)
# elem_cuda = torch.from_numpy(prep.selem).cuda().type(torch.float)
# t1 = perf_counter()
# dial = torch.nn.functional.conv3d(bin_lb_cuda.unsqueeze(0).unsqueeze(0),
#                                   elem_cuda.unsqueeze(0).unsqueeze(0), padding=(2, 15, 15))[0, 0]
# t2 = perf_counter() - t1
# print(t2)
# t1 = perf_counter()
# mask3 = (dial > 0).type(torch.float).cpu().numpy()
# t2 = perf_counter() - t1
# print(t2)
# # %%
