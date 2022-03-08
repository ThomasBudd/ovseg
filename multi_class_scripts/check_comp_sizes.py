from ovseg.data.Dataset import raw_Dataset
from ovseg.utils.label_utils import reduce_classes
import numpy as np
from skimage.measure import label
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import nibabel as nib

'''
Checking the sized of 2d connected components on axial slices
'''
ds = raw_Dataset('OV04_dcm')

comp_sizes = [[], []]

for i in tqdm(range(len(ds))):
    
    data_tpl = ds[i]
    
    if 'label' not in data_tpl:
        continue
    
    lb = reduce_classes(data_tpl['label'], [1, 9])
    
    for j in range(2):
        
        for z in range(lb.shape[0]):
            
            sl = (lb[z] == j+1).astype(int)
            if sl.max() > 0:
                
                comps = label(sl, connectivity=2)
                
                    
                comp_sizes[j].extend([np.sum(comps == c) for c in range(1, comps.max() + 1)])

comp_sizes = [np.array(cs) for cs in comp_sizes]
# %%

for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.hist(comp_sizes[i][comp_sizes[i] <= 100])

#%%
comp_sizes = [[], []]

predp = 'D:\\PhD\\Data\\ov_data_base\\predictions\\OV04\\pod_om_08_25\\U-Net4_prg_lrn\\BARTS_dcm_ensemble_5_6_7'

nii_files = [f for f in os.listdir(predp) if f.endswith('.nii.gz')]

for file in tqdm(nii_files):
    
    seg = nib.load(os.path.join(predp, file)).get_fdata()
    
    lb = reduce_classes(seg, [1, 9])
    
    for j in range(2):
        
        for z in range(lb.shape[0]):
            
            sl = (lb[z] == j+1).astype(int)
            if sl.max() > 0:
                
                comps = label(sl, connectivity=2)
                
                    
                comp_sizes[j].extend([np.sum(comps == c) for c in range(1, comps.max() + 1)])

comp_sizes = [np.array(cs) for cs in comp_sizes]

for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.hist(comp_sizes[i][comp_sizes[i] <= 100])
