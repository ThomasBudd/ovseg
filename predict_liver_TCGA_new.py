from ovseg.model.SegmentationModel import SegmentationModel
import os
from ovseg.utils.io import read_dcms
from tqdm import tqdm
import torch

model = SegmentationModel(val_fold=5, data_name='Lits_5mm',
                          preprocessed_name='default',
                          model_name='U-Net5')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'TCGA_new')
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 'Lits_5mm',
                     'default', 'U-Net5', 'TCGA_new_fold_5')
scans = []

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
    pred = model(data_tpl)
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()

    model.save_prediction(data_tpl, folder_name=predp, filename=scan)