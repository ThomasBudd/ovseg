import os
from ovseg.utils.io import read_dcms, read_nii
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

predp1 = 'D:\\PhD\\Data\\ov_data_base\\predictions\\OV04\\pod_om\\clara_model_no_tta\\ICON8_14_Derby_Burton_clara'
predp2 = 'D:\\PhD\\Data\\ov_data_base\\predictions\\ApolloTCGA_BARTS_OV04\\pod_om\\clara_model\\ICON8_14_Derby_Burton_clara'
plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'ICON8', 'clara_models')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'ICON8_14_Derby_Burton')
scans = os.listdir(rawp)

# %%
for scan in tqdm(scans):
    
    im = read_dcms(os.path.join(rawp, scan))['image'].clip(-150, 250)

    nii_file = [f for f in os.listdir(predp1) if f.startswith(scan) and f.endswith('.nii.gz')][0]
    p1, _, _ = read_nii(os.path.join(predp1,nii_file))
    p2, _, _ = read_nii(os.path.join(predp2,nii_file))
    
    
    contains = np.where(np.sum(p1+p2,(1,2)))[0]
    
    for z in contains:
        plt.imshow(im[z],cmap='bone')
        plt.contour(p1[z] > 0, colors='red')
        plt.contour(p2[z] > 0, colors='blue')
        plt.savefig(os.path.join(plotp, scan+'_'+str(z)), bbox_inches='tight')
        plt.close()
    