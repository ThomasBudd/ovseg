import numpy as np
from os import environ, listdir
from os.path import join
from skimage.measure import label
import nibabel as nib
from time import sleep
from tqdm import tqdm

w_list = [0.001, 0.01, 0.1]

gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'labels')

for w in w_list:
    predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'SLDS',
                 'U-Net5_'+str(w), 'cross_validation')
    n_reg = 0
    n_fg_reg = 0
    sleep(0.5)
    for case in tqdm(listdir(predp)):
        gt = (nib.load(join(gtp, case)).get_fdata() > 0).astype(float)
        regs = (nib.load(join(predp, case)).get_fdata() == 2).astype(float)
        comps = label(regs)
        n_reg += comps.max()
        for c in range(1, comps.max() + 1):
            comp = (comps == c).astype(float)
            n_fg_reg += (comp * gt).max()
    
    print('w:{}, produces {} regions of which {:.2f}% showed fg'.format(w,
                                                                        n_reg,
                                                                        100 * n_fg_reg/n_reg))
