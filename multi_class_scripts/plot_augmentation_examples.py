import matplotlib.pyplot as plt
import torch
import numpy as np
from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import os

plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'examples_augmentation')
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

for batch in model.data.trn_dl:
    break

batch = batch.cuda().type(torch.float)
batches = [batch[..., 160:-160, 160:-160]]

for _ in range(3):
    batch_aug = model.augmentation.torch_augmentation(torch.clone(batch))
    batches.append(batch_aug)

batches = [b.cpu().numpy() for b in batches]


# %%

for i, batch in enumerate(batches):
    for b in range(2):
        sample = batch[b]
        plt.imshow(sample[0, 20], cmap='bone')
        plt.contour(sample[1, 20] > 0, linewidths=0.5, colors='red')
        plt.axis('off')
        plt.savefig(os.path.join(plotp, 'aug_{}_{}'.format(b, i)), bbox_inches='tight')
        plt.close()

