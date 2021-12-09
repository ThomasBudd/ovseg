import matplotlib.pyplot as plt
import torch
import numpy as np
from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import os

plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'examples_sliding_window')
if not os.path.exists(plotp):
    os.makedirs(plotp)

patch_size = [40, 320, 320]
z_to_xy_ratio = 5.0/0.67
model_params = get_model_params_3d_res_encoder_U_Net(patch_size,
                                                     z_to_xy_ratio,
                                                     n_fg_classes=2)

model = SegmentationModel(val_fold=0, data_name='OV04_test',
                          preprocessed_name='pod_om_067',
                          model_name='plot_augmentation',
                          model_parameters=model_params)

# %%
w = model.prediction.patch_weight.cpu().numpy()

plt.subplot(1, 2, 1)
plt.imshow(w[0, 20], cmap='gray')
plt.axis('off')

# %%
plt.subplot(1, 2, 2)
data_tpl = model.data.trn_ds[1]
im = data_tpl['image'][0].astype(float)
xyz_list = model.prediction._get_xyz_list(im.shape)
xy_list = [(x, y) for z, x, y in xyz_list if z == 0]
print(xy_list)
def print_patch(x, y, col):
    xp = min([x+320, 511])
    yp = min([y+320, 511])
    plt.plot([x, xp], [y, y], col)
    plt.plot([xp, xp], [y, yp], col)
    plt.plot([xp, x], [yp, yp], col)
    plt.plot([x, x], [yp, y], col)

colors = ['red', 'green', 'blue', 'magenta', 'lime', 'cyan']

plt.imshow(im[50], cmap='bone')
for i, (x,y) in enumerate(xy_list[::2]):
    col = colors[i % len(colors)]
    print_patch(x, y, col)
plt.axis('off')
plt.savefig(os.path.join(plotp, 'sliding_window'), bbox_inches='tight')
plt.close()
