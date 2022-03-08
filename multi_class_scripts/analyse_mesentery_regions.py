import nibabel as nib
import numpy as np
from os import listdir, environ
from os.path import join
from tqdm import tqdm

gtp1 = join(environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'labels')
gtp2 = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')

predp1 = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'small_reg_expert', 'U-Net2_32',
              'cross_validation')
predp2 = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'small_reg_expert', 'U-Net2_32',
              'BARTS_ensemble_0_1_2_3_4')

lb_classes=[3,4,5,6,7,11,12,13,14,15,16,17,18]

# %%
has_fg = []
has_reg = []
sens = [[] for _ in range(len(lb_classes))]

for case in tqdm(listdir(gtp1)):
    
    gt = nib.load(join(gtp1, case)).get_fdata()
    pred = (nib.load(join(predp1, case)).get_fdata() > 0).astype(float)
    
    
    for i, cl in enumerate(lb_classes):
        lb = (gt == cl).astype(float)
        if lb.max() > 0:
            sens[i].append(100 * np.sum(lb * pred) / np.sum(lb))
    

print()    
print('OV04', ', '.join(['{}: {:.2f}'.format(cl, np.mean(s)) for cl, s in zip(lb_classes, sens)]))
# %%
has_fg2 = []
has_reg2 = []
sens2 = [[] for _ in range(len(lb_classes))]

for case in tqdm(listdir(gtp2)):
    
    gt = nib.load(join(gtp2, case)).get_fdata()
    pred = (nib.load(join(predp2, case)).get_fdata() > 0).astype(float)
    
    for i, cl in enumerate(lb_classes):
        lb = (gt == cl).astype(float)
        if lb.max() > 0:
            sens2[i].append(100 * np.sum(lb * pred) / np.sum(lb))
    
print()
print('BARTS', ', '.join(['{}: {:.2f}'.format(cl, np.mean(s)) for cl, s in zip(lb_classes, sens2)]))
