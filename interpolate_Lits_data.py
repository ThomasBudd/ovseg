import numpy as np
import nibabel as nib
import os
import torch
from tqdm import tqdm
from scipy.ndimage import zoom

rawp = os.path.join(os.environ['OV_DATA_BASE'],
                    'raw_data',
                    'Lits')
tp = os.path.join(os.environ['OV_DATA_BASE'],
                  'raw_data',
                  'Lits_5mm')

scans = os.listdir(os.path.join(rawp, 'labels'))

for f in ['images', 'labels']:
    if not os.path.exists(os.path.join(tp, f)):
        os.makedirs(os.path.join(tp, f))

for scan in tqdm(scans):
    
    im_scan = [s for s in os.listdir(os.path.join(rawp, 'images')) if
               s.startswith(scan.split('.')[0])][0]
    seg = nib.load(os.path.join(rawp, 'labels', scan))
    img = nib.load(os.path.join(rawp, 'images', im_scan))
    
    sp = seg.header['pixdim'][1:4]
    print(sp)
    if sp[2] == 5.0:
        continue
    
    sp_new = np.array([sp[0], sp[1], 5])
    
    nz = seg.shape[2]
    nz_new = int(nz * sp[2] / 5.0)
    
    sg = seg.get_fdata()
    im = img.get_fdata()
    
    # sgt = torch.from_numpy(sg[np.newaxis, np.newaxis]).cuda()
    # imt = torch.from_numpy(im[np.newaxis, np.newaxis]).cuda()
    
    # sg_new = torch.nn.functional.interpolate(sgt, [seg.shape[0], seg.shape[1], nz_new],
    #                                          mode='nearest').cpu().numpy()[0,0]
    # im_new = torch.nn.functional.interpolate(imt, [im.shape[0], im.shape[1], nz_new],
    #                                          mode='trilinear').cpu().numpy()[0,0]
    
    sg_new = zoom(sg, (1, 1, sp[2]/5), order=0)
    im_new = zoom(im, (1, 1, sp[2]/5), order=1)
    print(sg.shape, sg_new.shape)
    im_nii = nib.Nifti1Image(im_new, np.eye(4))
    im_nii.header['pixdim'][1:4] = sp_new
    
    sg_nii = nib.Nifti1Image(sg_new, np.eye(4))
    sg_nii.header['pixdim'][1:4] = sp_new
    
    nib.save(im_nii, os.path.join(tp, 'images', im_scan))
    nib.save(sg_nii, os.path.join(tp, 'labels', scan))

# %%
scans = os.listdir(os.path.join(tp, 'labels'))

for scan in scans:
    
    im_scan = [s for s in os.listdir(os.path.join(rawp, 'images')) if
               s.startswith(scan.split('.')[0])][0]
    seg = nib.load(os.path.join(tp, 'labels', scan))
    img = nib.load(os.path.join(tp, 'images', im_scan))
    print(img.shape, seg.shape)
