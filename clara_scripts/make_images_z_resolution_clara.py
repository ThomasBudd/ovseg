import numpy as np
import matplotlib.pyplot as plt
import os
from ovseg.utils.io import read_dcms, read_nii
from tqdm import tqdm
import pydicom

predp_clara = 'D:\\PhD\\Data\\ov_data_base\\predictions\\OV04\\pod_om\clara_model_no_tta\\ICON8_14_Derby_Burton_clara'
predp_ens = 'D:\\PhD\\Data\\ov_data_base\\predictions\\OV04\\pod_om\\clara_model_no_tta\\ICON8_14_Derby_Burton_ensemble_5_6_7'
rawp = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\ICON8_14_Derby_Burton'

# %% get scans with thin slices
thin_slices = []

for scan in tqdm(os.listdir(rawp)):
    
    dcms = [dcm for dcm in os.listdir(os.path.join(rawp, scan)) if dcm.endswith('.dcm')]
    ds_list = [pydicom.dcmread(os.path.join(rawp, scan, dcm)) for dcm in dcms]
    z_im = [ds.ImagePositionPatient[2] for ds in ds_list]
    diff = np.diff(np.sort(z_im))
    if np.median(diff) < 5.0:
        thin_slices.append(scan)

# %%
scan = thin_slices[0]
data_tpl = read_dcms(os.path.join(rawp, scan))
nii_file = [f for f in os.listdir(predp_clara) if f.startswith(scan) and f.endswith('.nii.gz')][0]
pred_clara, _, _ = read_nii(os.path.join(predp_clara, nii_file))
pred_ens, _, _ = read_nii(os.path.join(predp_ens, nii_file))

# %%
z = 76
xmn, xmx = 90, 150
ymn, ymx = 180, 240
im = data_tpl['image'].clip(-50,150)
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.imshow(im[z+i,xmn:xmx,ymn:ymx],cmap='bone')
    plt.contour(pred_ens[z+i,xmn:xmx,ymn:ymx] > 0, colors='red')
    plt.axis('off')
    plt.subplot(2,3,i+4)
    plt.imshow(im[z+i,xmn:xmx,ymn:ymx],cmap='bone')
    plt.contour(pred_clara[z+i,xmn:xmx,ymn:ymx] > 0, colors='red')
    plt.axis('off')
# %%
z = 119
xmn, xmx = 80, 170
ymn, ymx = 180, 270
im = data_tpl['image'].clip(-50,150)
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.imshow(im[z+i,xmn:xmx,ymn:ymx],cmap='bone')
    plt.contour(pred_ens[z+i,xmn:xmx,ymn:ymx] > 0, colors='red')
    plt.axis('off')
    if i == 0:
        plt.ylabel('conventional')
    plt.subplot(2,3,i+4)
    plt.imshow(im[z+i,xmn:xmx,ymn:ymx],cmap='bone')
    plt.contour(pred_clara[z+i,xmn:xmx,ymn:ymx] > 0, colors='red')
    plt.axis('off')
    if i == 0:
        plt.ylabel('dynamic z inference')

# %%

plotp = 'D:\\PhD\\Data\\ov_data_base\\plots\\dynamic_z_inference'
for i in range(3):
    plt.imshow(im[z+i,xmn:xmx,ymn:ymx],cmap='bone')
    plt.contour(pred_ens[z+i,xmn:xmx,ymn:ymx] > 0, colors='red')
    plt.axis('off')
    plt.savefig(os.path.join(plotp, 'conventional_'+str(i)), bbox_inches='tight')
    plt.close()
for i in range(3):
    plt.imshow(im[z+i,xmn:xmx,ymn:ymx],cmap='bone')
    plt.contour(pred_clara[z+i,xmn:xmx,ymn:ymx] > 0, colors='red')
    plt.axis('off')
    plt.savefig(os.path.join(plotp, 'dynamic_z_inference_'+str(i)), bbox_inches='tight')
    plt.close()