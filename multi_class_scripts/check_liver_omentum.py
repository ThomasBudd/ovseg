from skimage.measure import label
import numpy as np
import nibabel as nib
from os import listdir, environ, mkdir
from os.path import join, exists
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.pyplot as plt

predp = join(environ['OV_DATA_BASE'], 'predictions', 'Lits_5mm', 'default',
             'U-Net5')
rawp = join(environ['OV_DATA_BASE'], 'raw_data')

folders = ['BARTS_fold_5', 'OV04_fold_5']

plotp = 'D:\\PhD\\Data\\ov_data_base\\plots\\Lits_5mm\\default\\z_largest'

# %%

z_liver_list = []
z_om_list = []
om_vol_below_z_liver_list = []
om_vol_list = []

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
        z_liver = np.argmax(np.sum(liver, (0, 1)))
        z_liver_list.append(z_liver)
        
        om_vol_list.append(np.sum(om))
        om_vol_below_z_liver_list.append(np.sum(om[..., z_liver:]))
        
        
        im = nib.load(join(rawp, fol.split('_')[0], 'images', file.split('.')[0]+'_0000.nii.gz')).get_fdata()

        plt.imshow(im[..., z_liver], cmap='bone')
        plt.axis('off')
        plt.savefig(join(plotp, file.split('.')[0]))
        plt.close()
z_liver_list = np.array(z_liver_list)
z_om_list = np.array(z_om_list)
om_vol_below_z_liver_list = np.array(om_vol_below_z_liver_list)
om_vol_list = np.array(om_vol_list)
# %%
diff = z_liver_list - z_om_list

print(np.mean(z_liver_list < z_om_list))

print(np.mean(om_vol_below_z_liver_list) / np.mean(om_vol_list))