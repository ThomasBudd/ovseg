from skimage.measure import label
import numpy as np
import nibabel as nib
from os import listdir, environ, mkdir
from os.path import join, exists
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from ovseg.utils.seg_fg_dial import seg_eros, seg_fg_dial

predp = join(environ['OV_DATA_BASE'], 'predictions', 'Lits_5mm', 'default',
             'U-Net5')
rawp = join(environ['OV_DATA_BASE'], 'raw_data')

folders = ['BARTS_fold_5', 'OV04_fold_5']
datasets = ['BARTS', 'OV04']
# %%
def compute_r1_r2(liver, lesions, z_to_xy_ratio):
    
    # dial and erosion want z axis first
    liver = np.moveaxis(liver, -1, 0)
    lesions = np.moveaxis(lesions, -1, 0)
    
    # first r1
    vol_les = np.sum(lesions)
    
    r1 = 0
    
    ovlp = np.sum(lesions * liver)
    if ovlp < vol_les:
        
        while ovlp < vol_les and r1 < 30:
            r1 += 1
            liver_dial = seg_fg_dial(liver, r1, z_to_xy_ratio=z_to_xy_ratio, use_3d_ops=True)
            ovlp = np.sum(liver_dial * lesions)
    
    r2 = 0
    
    ovlp = np.sum(lesions * liver)
    if ovlp > 0:
        while ovlp > 0 and r2 < 30:
            r2 += 1
            liver_eros = seg_eros(liver, r2, z_to_xy_ratio=z_to_xy_ratio, use_3d_ops=True)
            ovlp = np.sum(liver_eros * lesions)
        
    return r1, r2

def get_z_min_max(liver, lesions):
    
    z_liver = np.where(np.sum(liver, (0,1)))[0]
    z_lesions = np.where(np.sum(lesions, (0, 1)))[0]
    
    return z_liver.min(), z_liver.max(), z_lesions.min(), z_lesions.max()

# %%

r_collection = {ds : [] for ds in datasets}
z_collection = {ds : [] for ds in datasets}

for ds in datasets:
    
    print(ds)
    sleep(0.1)
    
    lbp = join(rawp, ds, 'labels')
    
    for file in tqdm(listdir(join(predp, ds+'_fold_5'))):
        
        lesions = (nib.load(join(lbp, file)).get_fdata() == 2).astype(float)
        if lesions.max() == 0:
            continue
        
        img = nib.load(join(predp, ds+'_fold_5', file))
        z_to_xy_ratio = img.header['pixdim'][3]/img.header['pixdim'][1]
        liver = img.get_fdata()
        
        r_collection[ds].append(compute_r1_r2(liver, lesions, z_to_xy_ratio))
        
        # z_collection[ds].append(get_z_min_max(liver, lesions))
    
    r_collection[ds] = np.array(r_collection[ds])
    # z_collection[ds] = np.array(z_collection[ds])

np.save(join(environ['OV_DATA_BASE'], 'liver_RUQ_r1_r2.npy'), r_collection)

# %%
r_collection = np.load(join(environ['OV_DATA_BASE'], 'liver_RUQ_r1_r2.npy'))

print('BARTS')
print(np.max(r_collection['BARTS'], 0))
print('OV04')
print(np.max(r_collection['OV04'], 0))
# Z = z_collection['OV04']
# Z1 = Z[:, 0] < Z[:, 2]
# Z2 = Z[:, 1] > Z[:, 3]