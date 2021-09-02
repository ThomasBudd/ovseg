from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int)
# parser.add_argument("p", type=int)
args = parser.parse_args()

model_name = 'res_encoder'
use_prg_trn = False
patch_size = [32, 176, 176]
out_shape = [[20, 112, 112],
             [24, 128, 128],
             [28, 152, 152],
             [32, 176, 176]]
        

prg_trn_sizes = np.array(out_shape)
prg_trn_sizes[:, 1:] *= 2
model_params = get_model_params_3d_nnUNet(patch_size, 2,
                                          use_prg_trn=use_prg_trn,
                                          n_fg_classes=2)

del model_params['network']['kernel_sizes']
del model_params['network']['kernel_sizes_up']
del model_params['network']['n_pyramid_scales']
model_params['architecture'] = 'unetresencoder'
model_params['network']['block'] = 'res'
model_params['network']['z_to_xy_ratio'] = 5
model_params['network']['stochdepth_rate'] = 0

prg_trn_aug_params = {}
c = 4

prg_trn_aug_params['mm_var_noise'] = np.array([[0, 0.1/c], [0, 0.1]])
prg_trn_aug_params['mm_sigma_blur'] = np.array([[0.5/c, 0.5 + 1/c], [0.5, 1.5]])
prg_trn_aug_params['mm_bright'] = np.array([[1 - 0.3/c, 1 + 0.3/c], [0.7, 1.3]])
prg_trn_aug_params['mm_contr'] = np.array([[1 - 0.35/c, 1 + 0.5/c], [0.65, 1.5]])
prg_trn_aug_params['mm_low_res'] = np.array([[1, 1 + 1/c], [1, 2]])
prg_trn_aug_params['mm_gamma'] = np.array([[1 - 0.3/c, 1 + 0.5/c], [0.7, 1.5]])
prg_trn_aug_params['out_shape'] = out_shape
if use_prg_trn:
    model_params['training']['prg_trn_sizes'] = prg_trn_sizes
    model_params['training']['prg_trn_aug_params'] = prg_trn_aug_params
    model_params['training']['prg_trn_resize_on_the_fly'] = False
model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
model_params['training']['lr_params'] = {'n_warmup_epochs': 50, 'lr_max': 0.02}
model_params['training']['opt_params'] = {'momentum': 0.99,
                                          'weight_decay': 3e-5,
                                          'nesterov': True,
                                          'lr': 2*10**-2}

p_name = 'pod_om_10'
model = SegmentationModel(val_fold=args.gpu,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()
model.eval_raw_data_npz('BARTS')
model.clean()


# %%
# import matplotlib.pyplot as plt
# import torch
# for batch in model.data.trn_dl:
#     break
# batch = batch.cuda().type(torch.float)[..., 88: 264, 88:264]
# with torch.no_grad():
#     pred = model.network(batch[:, :1])
# sm = torch.nn.functional.softmax(pred[0], 1).cpu().numpy()
# pred = pred[0].cpu().numpy()
# batch = batch.cpu().numpy()

# for i in range(2):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(batch[0, i, 16], cmap='gray')
# for i in range(3):
#     plt.subplot(2, 5, i+3)
#     plt.imshow(sm[0, i, 16], cmap='gray')
# for i in range(2):
#     plt.subplot(2, 5, i+6)
#     plt.imshow(batch[1, i, 16], cmap='gray')
# for i in range(3):
#     plt.subplot(2, 5, i+8)
#     plt.imshow(sm[1, i, 16], cmap='gray')