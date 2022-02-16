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
plotp = join(environ['OV_DATA_BASE'], 'plots', 'Lits_5mm', 'default',
             'U-Net5')
rawp = join(environ['OV_DATA_BASE'], 'raw_data')

folders = ['BARTS_fold_5', 'OV04_fold_5']

# %%
def keep_only_largest_component(seg):
    
    ccs = label(seg)
    
    if ccs.max() < 2:
        largest = seg
    else:
        volumes = [np.sum(ccs == i) for i in range(1, ccs.max() + 1)]
        
        k = np.argmax(volumes)
        
        largest = (ccs == k+1).astype(seg.dtype)

    return np.stack([binary_fill_holes(largest[..., z]) for z in range(largest.shape[-1])], -1)

def plot_example(seg, fol, file):
    
    imp = join(rawp, fol.split('_')[0], 'images')
    nii_file = join(imp, file.split('.')[0] + '_0000.nii.gz')
    im = nib.load(nii_file).get_fdata().clip(-150, 250)
    
    z_list = np.where(np.sum(seg, (0, 1)))[0]
    
    if not exists(join(plotp, fol+'_cleaned')):
        mkdir(join(plotp, fol+'_cleaned'))

    for z in z_list:
        plt.imshow(im[..., z], cmap='bone')
        plt.contour(seg[..., z], colors='red')
        plt.axis('off')
        plt.savefig(join(plotp, fol+'_cleaned', file.split('.')[0] + '_' + str(int(z))+'.png'))
        plt.close()

# %%
for fol in folders:
    
    print(fol)
    sleep(0.1)
    for file in tqdm(listdir(join(predp, fol))):
        
        nii_file = join(predp, fol, file)
        
        img = nib.load(nii_file)
        
        seg = keep_only_largest_component(img.get_fdata())
        
        nib.save(nib.Nifti1Image(seg, img.affine, img.header), nii_file)
        
        # plot_example(seg, fol, file)