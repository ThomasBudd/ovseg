import os
from ovseg.utils.io import read_dcms, read_nii, save_nii
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
import nibabel as nib

rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'TCGA_new')
plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'Lits_5mm',
                     'default', 'U-Net5', 'TCGA_new_fold_5')
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 'Lits_5mm',
                     'default', 'U-Net5', 'TCGA_new_fold_5')
scans = []
if not os.path.exists(plotp):
    os.makedirs(plotp)

def keep_only_largest_component(seg):
    
    ccs = label(seg)
    
    if ccs.max() < 2:
        largest = seg
    else:
        volumes = [np.sum(ccs == i) for i in range(1, ccs.max() + 1)]
        
        k = np.argmax(volumes)
        
        largest = (ccs == k+1).astype(seg.dtype)

    return np.stack([binary_fill_holes(largest[..., z]) for z in range(largest.shape[-1])], -1)


for scan in os.listdir(rawp):
    
    scanp = os.path.join(rawp, scan)
    if 'abdomen' in os.listdir(scanp):
        
        scans.append(os.path.join(scanp, 'abdomen'))
    
    else:
        scans.append(scanp)

for i in tqdm(range(len(scans))):
    # get the data
    data_tpl = read_dcms(scans[i])
    # first let's try to find the name
    
    scan = data_tpl['pat_id']+'_liver_TB'
    
    # predict from this datapoint
    nii_file = os.path.join(predp, scan+'.nii.gz')
    img = nib.load(nii_file)
    liver = img.get_fdata()
    if i == 26:
        liver[77:] = 0
    liver = keep_only_largest_component(img.get_fdata()).astype(int)
        
    nib.save(nib.Nifti1Image(liver, img.affine, img.header), nii_file)
    
    z_list = np.where(np.sum(liver, (1, 2)))[0]
    
    for z in z_list:
        
        plt.imshow(data_tpl['image'][z].clip(-150, 250), cmap='bone')
        plt.contour(liver[z])
        plt.axis('off')
        plt.savefig(os.path.join(plotp, scan + '_'+str(z)), bbox_inches='tight')
        plt.close()
    