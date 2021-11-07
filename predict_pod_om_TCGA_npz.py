from ovseg.model.SegmentationModel import SegmentationModel
import os
from ovseg.utils.io import read_dcms, read_nii
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

data_name='ApolloTCGA_dcm_BARTS_dcm_OV04_dcm'
p_name='pod_om_08_25'
model_name='U-Net4_prg_lrn'
model = SegmentationModel(val_fold=args.vf,
                          data_name=data_name,
                          preprocessed_name=p_name,
                          model_name=model_name,
                          is_inference_only=True)
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'TCGA_new')
predp = os.path.join(os.environ['OV_DATA_BASE'], 'npz_predictions',
                     data_name,
                     p_name,
                     model_name,
                     'TCGA_new',
                     'fold_'+str(args.vf))
if not os.path.exists(predp):
    
    os.makedirs(predp)

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
    
    # it doesn't matter what how we set this value
    # it is just to not let the code crash in the ensemble prediction
    data_tpl['scan'] = scan
    
    # predict from this datapoint
    pred = model(data_tpl, do_postprocessing=False)
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()

    np.save(os.path.join(predp, scan), pred)