import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
plt.close('all')

raw_path = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'BARTS')
raw_imp = os.path.join(raw_path, 'images')
raw_lbp = os.path.join(raw_path, 'labels')
fbp_conv_path = 'D:\\PhD\\Data\\ov_data_base\\learned_recons\\fbp_conv_win'
LPD_path = 'D:\\PhD\\Data\\ov_data_base\\learned_recons\\fbp_conv_HU'

titels = ['Siemens', 'win sim', 'HU sim']
x, y, dx, dy = 0, 0, 512, 512
for case in os.listdir(fbp_conv_path):
    fbp_pred = nib.load(os.path.join(fbp_conv_path, case)).get_fdata()
    LPD_pred = nib.load(os.path.join(LPD_path, case)).get_fdata()
    gt = nib.load(os.path.join(raw_imp, case[:8]+'_0000.nii.gz')).get_fdata()
    lb = nib.load(os.path.join(raw_lbp, case)).get_fdata()
    lb = (lb > 0).astype(float)

    z_list = [np.argmax(np.sum(lb, (0, 1))), np.random.randint(lb.shape[-1])]

    print('{}, {}, {}'.format(*[vol.shape for vol in [gt, fbp_pred, LPD_pred]]))
    plt.figure()
    for i, z in enumerate(z_list):
        for j, vol in enumerate([gt, fbp_pred, LPD_pred]):
            plt.subplot(2, 3, i*3 + j + 1)
            plt.imshow(vol[x:x+dx, y:y+dy, z].clip(-50, 250), cmap='gray')
            plt.contour(lb[x:x+dx, y:y+dy, z], linewidths=0.5, colors='red', linestyles='solid')
            if i == 0:
                plt.title(titels[j])
# %%
x, y, dx, dy = 128, 128, 128, 128
for case in os.listdir(fbp_conv_path):
    fbp_pred = nib.load(os.path.join(fbp_conv_path, case)).get_fdata()
    LPD_pred = nib.load(os.path.join(LPD_path, case)).get_fdata()
    gt = nib.load(os.path.join(raw_imp, case[:8]+'_0000.nii.gz')).get_fdata()
    lb = nib.load(os.path.join(raw_lbp, case)).get_fdata()
    lb = (lb > 0).astype(float)

    z_list = 2* [np.argmax(np.sum(lb, (0, 1)))]

    print('{}, {}, {}'.format(*[vol.shape for vol in [gt, fbp_pred, LPD_pred]]))
    plt.figure()
    for i, z in enumerate(z_list):
        for j, vol in enumerate([gt, fbp_pred, LPD_pred]):
            plt.subplot(2, 3, i*3 + j + 1)
            plt.imshow(vol[x:x+dx, y:y+dy, z].clip(-50, 250), cmap='gray')
            if i == 0:
                plt.contour(lb[x:x+dx, y:y+dy, z], linewidths=0.5, colors='red', linestyles='solid')
                plt.title(titels[j])
