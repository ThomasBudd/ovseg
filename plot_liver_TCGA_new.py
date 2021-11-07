import os
from ovseg.utils.io import read_dcms, read_nii
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label

rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'TCGA_new')
plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'Lits_5mm',
                     'default', 'U-Net5', 'TCGA_new_fold_5')
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 'Lits_5mm',
                     'default', 'U-Net5', 'TCGA_new_fold_5')
scans = []
if not os.path.exists(plotp):
    os.makedirs(plotp)

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
    liver = read_nii(os.path.join(predp, scan+'.nii.gz'))[0]
    
    comps = label(liver)
    
    if comps.max() == 0:
        print('no liver segmented!')
        continue
    
    if comps.max() > 1:
        print('keeping only largest component didn\'t work!')
    
    z_list = np.where(np.sum(liver, (1,2)))[0]
    
    for z in z_list:
        
        plt.imshow(data_tpl['image'][z], cmap='bone')
        plt.contour(liver[z])
        plt.axis('off')
        plt.savefig(os.path.join(plotp, scan + '_'+str(z)), bbox_inches='tight')
        plt.close()
    