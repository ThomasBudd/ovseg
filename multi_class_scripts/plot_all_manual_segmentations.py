import os
import numpy as np
import matplotlib.pyplot as plt
from ovseg.utils.io import read_dcms
from tqdm import tqdm

VTT_path = 'D:\PhD\Data\VTT'
plotp = os.path.join(VTT_path, 'plots')
if not os.path.exists(plotp):
    os.mkdir(plotp)


def get_roi_dcms(dcmp):
    return [os.path.join(dcmp, dcm) for dcm in os.listdir(dcmp)
            if dcm.startswith('TCGA') and dcm.endswith('.dcm')]

for task in os.listdir(VTT_path):
    if task == 'plots':
        continue
    
    scans = os.listdir(os.path.join(VTT_path, task))
    
    for scan in tqdm(scans):
        
        dcmp = os.path.join(VTT_path, task, scan)
        roi_dcm = get_roi_dcms(dcmp)[0]
        
        if roi_dcm.endswith('TB.dcm'):
            continue
        
        data_tpl = read_dcms(dcmp)
        
        im = data_tpl['image'].clip(-150, 250).astype(float)
        lb = (data_tpl['label'] > 0).astype(float)
        
        contains = np.where(np.sum(lb, (1, 2)))[0]
        
        for z in contains:
            
            x, y = np.where(lb[z])
            xmn, xmx = np.max([x.min()-10, 0]), np.min([x.max()+11, 512])
            ymn, ymx = np.max([y.min()-10, 0]), np.min([y.max()+11, 512])
            plt.imshow(im[z, xmn:xmx, ymn:ymx], cmap='bone')
            plt.contour(lb[z, xmn:xmx, ymn:ymx], colors='red', linewidths=0.5)
            plt.axis('off')
            plt.savefig(os.path.join(plotp, '_'.join([task, scan, str(z)])))
            plt.close()
        