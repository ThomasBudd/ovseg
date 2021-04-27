import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

dp = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\BARTS\\labels'
predp = 'D:\\PhD\\Data\\nnUnet_raw_data_base\\nnUNet_predictions\\120\\3d_lowres_predictions'

dscs = []
for case in tqdm(os.listdir(dp)):
    lb = (nib.load(os.path.join(dp, case)).get_fdata() == 9).astype(float)
    if lb.max() > 0:
        pred = nib.load(os.path.join(predp, case)).get_fdata()
        dscs.append(200 * np.sum(lb * pred) / np.sum(lb+pred))

# %%
import torch
from torch.cuda.amp import autocast

xb = torch.rand(1, 1, 100, 100).cuda()
print(xb.dtype)

conv = torch.nn.Conv2d(1, 1, 1).cuda()

with autocast():
    print(conv(xb).dtype)

    with autocast(enabled=False):
        print(conv(xb).dtype)

        