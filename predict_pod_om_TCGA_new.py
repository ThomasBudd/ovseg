from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import os
from ovseg.utils.io import read_dcms, read_nii
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt


data_name='ApolloTCGA_dcm_BARTS_dcm_OV04_dcm'
p_name='pod_om_08_25'
model_name='U-Net4_prg_lrn'
model = SegmentationEnsemble(data_name=data_name,
                             preprocessed_name=p_name,
                             model_name=model_name)
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'TCGA_new')
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions',
                     data_name,
                     p_name,
                     model_name,
                     'TCGA_new')
livp = os.path.join(os.environ['OV_DATA_BASE'],
                    'predictions',
                    'Lits_5mm',
                    'default',
                    'U-Net5')
plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots',
                     data_name,
                     p_name,
                     model_name,
                     'TCGA_new')
if not os.path.exists(plotp):
    
    os.makedirs(plotp)

scans = []

for root, dirs, files in os.walk(rawp):
    
    if len(files) > 0:
        
        scans.append(root)

for i in tqdm(range(len(scans))):
    
    scanp = scans[i]
    # get the data
    data_tpl = read_dcms(scanp)
    # first let's try to find the name
    
    scan = data_tpl['pat_id']
    
    if scanp.endswith('abdomen'):
        scan += '_abdomen'
    elif scanp.endswith('pelvis'):
        scan += '_pelvis'
    
    scan += '_TB'
    
    # predict from this datapoint
    pred = model(data_tpl)
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()

    if not scanp.endswith('pelvis'):
        # load the liver segmentation
        nii_file = os.path.join(livp, data_tpl['pat_id']+'_liver_TB')
        liver = read_nii(nii_file)
        
        # compute the slice where the liver is the largest
        z_liver = np.argmax(np.sum(liver, (1,2)))
        
        # plot this slice
        plt.imshow(data_tpl['image'][z_liver], cmap='bone')
        plt.contour(liver[z_liver])
        plt.contour(data_tpl[model.pred_key][z_liver])
        plt.axis('off')
        plt.savefig(os.path.join(plotp, data_tpl['pat_id']))
        plt.close()
        # set prediction above to 0
        data_tpl[model.pred_key][:z_liver] = 0

    model.save_prediction(data_tpl, folder_name=predp, filename=scan)