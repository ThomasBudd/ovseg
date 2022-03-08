from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
import matplotlib.pyplot as plt
import numpy as np

model_params = get_model_params_3d_nnUNet([32, 64, 64], 0, fp32=True)

model_params['training']['num_epochs'] = 100
model_params['network']['filters'] = 8
model_params['data']['trn_dl_params']['store_data_in_ram'] = True

model = SegmentationModel(val_fold=0, data_name='OV04', preprocessed_name='pod_quater',
                          model_name='test_3d_training', model_parameters=model_params)
# %%

for batch in model.data.trn_dl:
    break

im = batch[0, 0].cpu().numpy()
lb = batch[0, 1].cpu().numpy()
z = np.argmax(np.sum(lb, axis=(1, 2)))

plt.imshow(im[z], cmap='gray')
plt.contour(lb[z])

# %%
model.training.train()
model.eval_validation_set(save_preds=False)
model.eval_training_set(save_preds=False)


# %%
batch = batch.cuda()
batch_aug = model.augmentation.torch_augmentation(batch.detach().clone()).cpu().numpy()
im_aug, lb_aug = batch_aug[0]
z_aug = np.argmax(np.sum(lb_aug, (1, 2)))
im, lb = batch[0].cpu().numpy()
z = np.argmax(np.sum(lb, (1, 2)))
plt.subplot(1, 2, 1)
plt.imshow(im[z], cmap='gray')
plt.contour(lb[z])

plt.subplot(1, 2, 2)
plt.imshow(im_aug[z_aug], cmap='gray')
plt.contour(lb_aug[z_aug])
