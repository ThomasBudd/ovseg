from ovseg.model.ClassCascadeModel import ClassCascadeModel
import numpy as np
import matplotlib.pyplot as plt

from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net


patch_size = [40, 320, 320]
model_name = 'U-Net5'
use_prg_trn = False
out_shape = [[24, 192, 192],
             [28, 224, 224],
             [36, 288, 288],
             [40, 320, 320]]
larger_res_encoder = True
model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                      z_to_xy_ratio=5.0/0.67,
                                                      use_prg_trn=use_prg_trn,
                                                      larger_res_encoder=larger_res_encoder,
                                                      n_fg_classes=6,
                                                      out_shape=out_shape)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                           'dice_loss']}
model_params['network']['in_channels'] = 2
model_params['data']['folders'] = ['images', 'labels', 'prev_preds']
model_params['data']['keys'] = ['image', 'label', 'prev_pred']
model_params['training']['batches_have_masks'] = True
model_params['postprocessing'] = {'mask_with_reg': True}
model_params['data']['val_dl_params']['n_fg_classes'] = 3
model_params['data']['trn_dl_params']['n_fg_classes'] = 3


model = ClassCascadeModel(val_fold=0, data_name='OV04_test',
                          preprocessed_name='ClassSegmentation',
                          model_name='test_class_cascade',
                          model_parameters=model_params)

for batch in model.data.trn_dl:
    break
batch = batch.cpu().numpy().astype(float)

# %%
b = batch[1]
im = b[0, 20]
im = (im - im.min()) / (im.max() - im.min())
mask = b[2, 20]
rgb_im = np.stack([im, im + 0.1 * mask, im], -1) / 1.1
plt.imshow(rgb_im)
plt.contour(b[1, 20] > 0, colors='b')
plt.contour(b[3, 20] > 0, colors='r')
