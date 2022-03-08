import nibabel as nib
from os import listdir, environ
from os.path import join
import numpy as np
from tqdm import tqdm
from time import sleep
gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')

w_list = [0.1]#[0.001, 0.01, 0.1]

for w in w_list:
    
    dscs = []
    sens_15 = []
    p1 = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'SLDS', 'U-Net5_{}'.format(w),
              'BARTS_ensemble_0_1_2_3_4')
    p2 = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'SLDS_reg_expert_{}'.format(w),
              'U-Net2','BARTS_ensemble_0_1_2_3_4')
    
    sleep(0.5)
    for case in tqdm(listdir(p1)):
        gt = nib.load(join(gtp, case)).get_fdata()
        gt_bin = (gt > 0).astype(float)
        gt_15 = (gt == 15).astype(float)
        pred1 = (nib.load(join(p1, case)).get_fdata() == 1).astype(float)
        reg = (nib.load(join(p1, case)).get_fdata() == 2).astype(float)
        pred2 = (nib.load(join(p2, case)).get_fdata() > 0 ).astype(float)
        pred = pred1 + pred2 * reg
        
        if pred.max() > 1:
            print('We got overlap for case {}'.format(case))
        
        if gt_bin.max() > 0:
            dscs.append(200 * np.sum(gt_bin * pred) / np.sum(gt_bin + pred))
        
        if gt_15.max() > 0:
            sens_15.append(100 * np.sum(gt_15 * pred) / np.sum(gt_15 + pred))
    print('w: {}, mean dsc: {:.2f}, sens 15: {:.2f}'.format(w, np.mean(dscs), np.mean(sens_15)))


# %%
import matplotlib.pyplot as plt
imp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'images')
case = 'case_288.nii.gz'
im = nib.load(join(imp, case[:8]+'_0000.nii.gz')).get_fdata().clip(-150, 250)
gt = nib.load(join(gtp, case)).get_fdata()
gt_bin = (gt > 0).astype(float)
gt_15 = (gt == 15).astype(float)
pred1 = nib.load(join(p1, case)).get_fdata()
pred2 = nib.load(join(p2, case)).get_fdata()

s1 = (pred2 == 2).astype(float)
s2 = s1 * (pred1 > 0).astype(float)
# %%
z = 42

plt.imshow(im[...,z],cmap='bone')
plt.contour(gt_bin[..., z], colors='red', linewidths=0.5)
plt.contour(pred1[..., z] == 1, colors='blue', linewidths=0.5)
plt.contour(pred1[..., z] == 2, colors='pink', linewidths=0.5)
plt.contour(pred2[..., z] == 2, colors='green', linewidths=0.5)

# %%
coords = np.stack(np.where(gt_bin > 0))

x,y,z = coords[:, 0]

plt.imshow(im[x-24:x+24, y-24:y+24,z],cmap='bone')
plt.contour(gt_bin[x-24:x+24, y-24:y+24, z], colors='red', linewidths=0.5)