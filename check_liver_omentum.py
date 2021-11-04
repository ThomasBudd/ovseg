from skimage.measure import label
import numpy as np
import nibabel as nib
from os import listdir, environ, mkdir
from os.path import join, exists
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes

predp = join(environ['OV_DATA_BASE'], 'predictions', 'Lits_5mm', 'default',
             'U-Net5')
rawp = join(environ['OV_DATA_BASE'], 'raw_data')

folders = ['BARTS_fold_5', 'OV04_fold_5']

# %%

z_liver_list = []
z_om_list = []

for fol in folders:
    
    print(fol)
    sleep(0.1)
    for file in tqdm(listdir(join(predp, fol))):
        
        lesions = nib.load(join(rawp, fol.split('_')[0], 'labels', file)).get_fdata()
        om = (lesions == 1).astype(float)
        
        if om.max() == 0:
            continue
        
        contains_om = np.where(np.sum(om, (0, 1)))[0]
        z_om_list.append(contains_om.min())
        
        liver = nib.load(join(predp, fol, file)).get_fdata()
        z_liver_list.append(np.argmax(np.sum(liver, (0, 1))))

z_liver_list = np.array(z_liver_list)
z_om_list = np.array(z_om_list)