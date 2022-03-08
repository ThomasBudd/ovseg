import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from os import listdir, environ
from os.path import join
from skimage.measure import label
from tqdm import tqdm

gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')
predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'bin_seg', 'U-Net5_M_15',
             'BARTS_ensemble_0_1_2_3_4')

cases = listdir(gtp)
case = cases[0]

sens_list = []
vol_list = []
bbox_list = []
for case in tqdm(cases):    
    img = nib.load(join(gtp, case))
    sp = img.header['pixdim'][1:4]
    gt = (img.get_fdata() > 0).astype(float)
    pred = (nib.load(join(predp, case)).get_fdata() > 0).astype(float)
    
    lb = label(gt)
    
    
    for c in range(1, lb.max() + 1):
        comp = (lb == c).astype(float)
        vol = np.prod(sp) * np.sum(comp)
        bbox = int(np.max((np.stack(np.where(comp)) * sp[:, np.newaxis]).ptp(1))/0.67+0.5)
        sens = 100 * np.prod(sp) * np.sum( comp * pred) / vol
        sens_list.append(sens)
        vol_list.append(vol)
        bbox_list.append(bbox)


plt.plot(bbox_list, sens_list, 'bo')