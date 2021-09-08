import numpy as np
import torch
from ovseg.model.RestaurationModel import RestaurationModel
from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from os import listdir, environ
from os.path import join
import matplotlib.pyplot as plt

prep = join(environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_2d')
jointp = join(environ['OV_DATA_BASE'], 'trained_models', 'OV04', 'pod_2d',
              'joint_rest_seg_refine_quater_1.0')

cases = ['case_{}'.format(i) for i in range(276, 279)]


rest_model = RestaurationModel(val_fold=0,
                               data_name='OV04',
                               model_name='restauration_fbps_quater',
                               preprocessed_name='pod_2d',
                               dont_store_data_in_ram=True)


# load jointly trained restauration weights
path_to_weights = join(jointp, 'fold_5', 'reconstruction_weights')
rest_model.network.load_state_dict(torch.load(path_to_weights,
                                              map_location=torch.device('cuda')))
model_params = get_model_params_2d_segmentation()
del model_params['augmentation']['torch_params']['grayvalue']
model_params['network']['norm'] = 'inst'
model_name = 'delete_me_2d_joint_quater'

# this is just creating a new blank segmentation model with random weights
seg_model = SegmentationModel(val_fold=5,
                              data_name='OV04',
                              model_name=model_name,
                              model_parameters=model_params,
                              preprocessed_name='pod_2d',
                              dont_store_data_in_ram=True)

# load jointly trained segmentation weights
path_to_weights = join(jointp, 'fold_5', 'segmentation_weights')
seg_model.network.load_state_dict(torch.load(path_to_weights,
                                             map_location=torch.device('cuda')))

case = cases[0]
keys = ['image_gt', 'label', 'fbp', 'recon_seq']
folders = ['images', 'labels', 'fbps_quater', 'restaurations_quater']
data_tpl = {key: np.load(join(prep, f, case+'.npy')) for key, f in zip(keys, folders)}
fp = np.load(join(prep, 'fingerprints', case+'.npy'), allow_pickle=True).item()
data_tpl.update(fp)

rest_joint = rest_model(data_tpl)
data_tpl['image'] = rest_joint

pred_joint = seg_model(data_tpl)

z1 = np.argmax(np.sum(data_tpl['label'], (1,2)))
contains_gt = np.where(np.sum(data_tpl['label'], (1,2)))[0]
contains_pred = np.where(np.sum(pred_joint, (1,2)))[0]
contains_both = [z for z in contains_gt if z in contains_pred]
z2 = np.random.choice(contains_both)

# for i, z in enumerate([z1, z2]):
#     plt.subplot(2, 4, 1 + 4*i)
#     plt.imshow(data_tpl['image_gt'][z], cmap='bone')
#     plt.contour(data_tpl['label'][z], colors='red')
#     plt.contour(pred_joint[z], colors='blue')
#     plt.subplot(2, 4, 2 + 4*i)
#     plt.imshow(data_tpl['image_gt'][z], cmap='bone')
#     plt.subplot(2, 4, 3 + 4*i)
#     plt.imshow(rest_joint[z], cmap='bone')
#     plt.subplot(2, 4, 4 + 4*i)
#     plt.imshow(data_tpl['recon_seq'][z], cmap='bone')

# plt.savefig('~/example_joint.png')

# %%

ims_max = []
ims_rand= []

for j, case in enumerate(cases):
    data_tpl = {key: np.load(join(prep, f, case+'.npy')) for key, f in zip(keys, folders)}
    fp = np.load(join(prep, 'fingerprints', case+'.npy'), allow_pickle=True).item()
    data_tpl.update(fp)
    
    rest_joint = rest_model(data_tpl)
    data_tpl['image'] = rest_joint
    
    pred_joint = seg_model(data_tpl)
    
    z1 = np.argmax(np.sum(data_tpl['label'], (1,2)))
    contains_gt = np.where(np.sum(data_tpl['label'], (1,2)))[0]
    contains_pred = np.where(np.sum(pred_joint, (1,2)))[0]
    contains_both = [z for z in contains_gt if z in contains_pred and z != z1]
    z2 = np.random.choice(contains_both)

    z = z1

    ims_max.append(np.stack([data_tpl['image_gt'][z],
                             data_tpl['label'][z],
                             pred_joint[z],
                             rest_joint[z],
                             data_tpl['recon_seq'][z]]))
    
    z = z2
    
    ims_rand.append(np.stack([data_tpl['image_gt'][z],
                             data_tpl['label'][z],
                             pred_joint[z],
                             rest_joint[z],
                             data_tpl['recon_seq'][z]]))

ims_max = np.stack(ims_max).astype(np.float32)
ims_rand = np.stack(ims_rand).astype(np.float32)

np.save('ims_joint_max', ims_max)
np.save('ims_joint_rand', ims_rand)


# %%

ims_max = np.load('ims_joint_max')
ims_rand = np.load('ims_joint_rand')

plt.figure()
for j in len(ims_max):
    plt.subplot(3, 4, 1 + 4*j)
    plt.imhsow(ims_max[j, 0], cmap='bone')
    plt.contour(ims_max[j, 1], colors='red')
    plt.contour(ims_max[j, 2], colors='blue')
    plt.axis('off')
    plt.subplot(3, 4, 2 + 4*j)
    plt.imhsow(ims_max[j, 0], cmap='bone')
    plt.axis('off')
    plt.subplot(3, 4, 3 + 4*j)
    plt.imshow(ims_max[j, 3], cmap='bone')
    plt.axis('off')
    plt.subplot(3, 4, 4 + 4*j)
    plt.imshow(ims_max[j, 4], cmap='bone')
    plt.axis('off')

plt.figure()
for j in len(ims_rand):
    plt.subplot(3, 4, 1 + 4*j)
    plt.imhsow(ims_rand[j, 0], cmap='bone')
    plt.contour(ims_rand[j, 1], colors='red')
    plt.contour(ims_rand[j, 2], colors='blue')
    plt.axis('off')
    plt.subplot(3, 4, 2 + 4*j)
    plt.imhsow(ims_rand[j, 0], cmap='bone')
    plt.axis('off')
    plt.subplot(3, 4, 3 + 4*j)
    plt.imshow(ims_rand[j, 3], cmap='bone')
    plt.axis('off')
    plt.subplot(3, 4, 4 + 4*j)
    plt.imshow(ims_rand[j, 4], cmap='bone')
    plt.axis('off')
