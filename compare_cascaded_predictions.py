import matplotlib.pyplot as plt
from os import environ, listdir
from os.path import join
import numpy as np
from tqdm import tqdm
import nibabel as nib


p1p = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'pod_om_08_25',
           'U-Net4_prg_lrn', 'ApolloTCGA_dcm_ensemble_5_6_7')
p2p = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'pod_om_08_25',
           'U-Net4_prg_lrn', 'ApolloTCGA_dcm_fold_5')

plotp = join(environ['OV_DATA_BASE'], 'plots', 'OV04', 'pod_om_08_25',
             'U-Net4_prg_lrn', 'ensemble_comp')

nii_files = [f for f in listdir(p1p) if f.endswith('.nii.gz')]

for nii_file in tqdm(nii_files):
    
    pens = nib.load(join(p1p, nii_file)).get_fdata()
    psing = nib.load(join(p2p, nii_file)).get_fdata()

    contains = np.where(np.sum(pens + psing, (1, 2)))[0]
    
    for z in contains:
        
        plt.subplot(1, 2, 1)
        plt.imshow(pens[z] > 0, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(psing[z] > 0, cmap='gray')
        plt.axis('off')
        
        plt.savefig(join(plotp, nii_file.split('.')[0] + '_'+str(z)))
        plt.close()