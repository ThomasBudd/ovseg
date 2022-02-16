import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm

data_name = 'OV04'
p_name = 'pod_om_08_5'
model_name = 'CV_refine_0_1'

plotp = os.path.join(os.environ['OV_DATA_BASE'],
                     'plots',
                     data_name,
                     p_name,
                     model_name,
                     'BARTS')

if not os.path.exists(plotp):
    os.makedirs(plotp)

predp = os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     data_name,
                     p_name,
                     model_name,
                     'BARTS_ensemble_5_6_7')

rawp = os.path.join(os.environ['OV_DATA_BASE'],
                    'raw_data',
                    'BARTS',
                    'images')
scans = os.listdir(rawp)

# %%
for scan in tqdm(scans):
    im = nib.load(os.path.join(rawp, scan)).get_fdata().clip(-150, 250)
    pred = nib.load(os.path.join(predp, scan[:8]+'.nii.gz')).get_fdata()
    
    contains = np.where(np.sum(pred, (0,1)))[0]
    
    for z in contains:
        plt.imshow(im[..., z],cmap='bone')
        plt.contour(pred[..., z] > 0, colors='red')
        plt.axis('off')
        plt.savefig(os.path.join(plotp, scan[:8]+'_'+str(z)), bbox_inches='tight')
        plt.close()