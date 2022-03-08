from ovseg.preprocessing.SegmentationPreprocessing import torch_preprocessing, np_preprocessing
from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing

import numpy as np
import torch
import matplotlib.pyplot as plt
from ovseg.utils.io import read_dcms

from time import perf_counter

data_tpl = read_dcms('D:\\PhD\\Data\\NEW_Barts_segmentations_VB_RW\\ID_001_1')

xb = np.stack([data_tpl['image'], data_tpl['label']])[np.newaxis].astype(np.float32)

preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                          apply_pooling=True,
                                          apply_windowing=True,
                                          target_spacing=[5.0, 0.7, 0.7],
                                          pooling_stride=[1, 2, 2],
                                          window=[-150, 250],
                                          lb_classes=[7],
                                          lb_min_vol=50,
                                          scaling=[50, 25])

preprocessing_torch = torch_preprocessing(apply_resizing=True,
                                          apply_pooling=True,
                                          apply_windowing=True,
                                          target_spacing=[5.0, 0.7, 0.7],
                                          pooling_stride=[1, 2, 2],
                                          window=[-150, 250],
                                          lb_classes=[7],
                                          lb_min_vol=50,
                                          scaling=[50, 25])
# preprocessing_np = np_preprocessing(apply_resizing=True,
#                                     apply_pooling=True,
#                                     apply_windowing=True,
#                                     target_spacing=[5.0, 0.7, 0.7],
#                                     pooling_stride=[1, 2, 2],
#                                     window=[-150, 250],
#                                     lb_classes=[7],
#                                     lb_min_vol=50,
#                                     scaling=[50, 25])


xb_prep = preprocessing(data_tpl).cpu().numpy()
xb_prep_np = preprocessing.np_preprocessing(xb, data_tpl['spacing'])

# %%
c = 1
z = 39  # np.argmax(np.sum(data_tpl['label'] == c, (1, 2)))
plt.subplot(2, 3, 1)
plt.imshow(xb[0, 1, z] == 7, cmap='gray')
plt.subplot(2, 3, 4)
plt.imshow(xb[0, 0, z], cmap='gray')
plt.colorbar()
plt.subplot(2, 3, 2)
plt.imshow(xb_prep[1, z] == 1, cmap='gray')
plt.subplot(2, 3, 5)
plt.imshow(xb_prep[0, z], cmap='gray')
plt.colorbar()
plt.subplot(2, 3, 3)
plt.imshow(xb_prep_np[0, 1, z] == 1, cmap='gray')
plt.subplot(2, 3, 6)
plt.imshow(xb_prep_np[0, 0, z], cmap='gray')
plt.colorbar()

# %%
n_reps = 10
st_cuda = perf_counter()
for _ in range(n_reps):
    xb_cuda = torch.from_numpy(xb).type(torch.float).cuda()
    xb_cuda_prep = preprocessing.torch_preprocessing(xb_cuda, data_tpl['spacing'])
et_cuda = perf_counter()
print('Cuda preprocessing: {:.5f}s per data_tpl'.format((et_cuda - st_cuda) / n_reps))
st_cpu = perf_counter()
for _ in range(n_reps):
    xb_prep = preprocessing.np_preprocessing(xb, data_tpl['spacing'])
et_cpu = perf_counter()
print('Cpu preprocessing: {:.5f}s per data_tpl'.format((et_cpu - st_cpu) / n_reps))
