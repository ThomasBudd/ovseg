from os import rename, environ, listdir, makedirs
from os.path import join, exists
import nibabel as nib
from tqdm import tqdm
from time import sleep
import numpy as np
import torch
from skimage.transform import resize

rbp = join(environ['OV_DATA_BASE'], 'raw_data', 'Lits')
tp = join(environ['OV_DATA_BASE'], 'raw_data', 'Lits_5mm')


for fol, mode in zip(['images', 'labels'], ['trilinear', 'nearest']):
    print(fol)
    sleep(0.2)
    for case in tqdm(listdir(join(rbp, fol))):
        
        if exists(join(tp, fol, case)):
            img = nib.load(join(tp, fol, case))
            img.affine[2, 2] = 5.0
            img.header['pixdim'][3] = 5.0
            nib.save(img, join(tp, fol, case))
            continue
            
        img = nib.load(join(rbp, fol, case))
        z_sp = img.header['pixdim'][3]
        
        
        if z_sp == 5.0:
            nib.save(img, join(tp, fol, case))
            continue
        
        im = img.get_fdata()
        nx, ny, nz = img.shape
        nz_new = int(nz * z_sp / 5 + 0.5)
        
        
        try:
            inpt = torch.from_numpy(im[np.newaxis, np.newaxis]).cuda()
            
            im_rsz = torch.nn.functional.interpolate(inpt, size=(nx, ny, nz_new),
                                                      mode=mode)[0,0].cpu().numpy()
        except:
            order = 1 if 'trilinear' else 0
            
            im_rsz = resize(im, np.array((nx, ny, nz_new)), order=order)
        img_new = nib.Nifti1Image(im_rsz, img.affine, img.header)
        img_new.affine[2, 2] = 5.0
        img_new.header['pixdim'][3] = 5.0
        nib.save(img_new, join(tp, fol, case))
        
        