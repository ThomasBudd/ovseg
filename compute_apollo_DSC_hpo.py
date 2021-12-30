import numpy as np
from ovseg.utils.io import load_pkl
import os
import nibabel as nib
from tqdm import tqdm

path_to_di = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
                          'ApolloTCGA', 'data_info.pkl')
di = load_pkl(path_to_di)
path_to_results = os.path.join(os.environ['OV_DATA_BASE'],
                               'trained_models',
                               'OV04',
                               'pod_om_08_5',
                               'U-Net4_prg_lrn',
                               'ensemble_0_1_2_3_4',
                               'ApolloTCGA_results.pkl')
results = load_pkl(path_to_results)

dices_1 = []
dices_9 = []

for key in results:
    if di[key[5:]]['dataset'] == 'Apollo':
        dices_1.append(results[key]['dice_1'])
        dices_9.append(results[key]['dice_9'])

print('my network')
print('omentum:')
print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(dices_1), np.nanmedian(dices_1)))
print('pelvic/ovarian:')
print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(dices_9), np.nanmedian(dices_9)))

# %% now compute scores for nnU-Net
predp = 'D:\\PhD\\Data\\nnUnet_raw_data_base\\nnUNet_predictions'
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
                          'BARTS', 'labels')

DSC9 = []
DSC1 = []

for case in tqdm(os.listdir(rawp)):
    gt = nib.load(os.path.join(rawp, case)).get_fdata()
    pod = (gt == 9).astype(float)
    om = (gt == 1).astype(float)
    if pod.max() > 0:
        
        pred = nib.load(os.path.join(predp, '120', case)).get_fdata()
        DSC9.append(200 * np.sum(pod*pred) / np.sum(pod+pred))
    
    if om.max() > 0:
        
        pred = nib.load(os.path.join(predp, '121', case)).get_fdata()
        DSC1.append(200 * np.sum(om*pred) / np.sum(om+pred))

print('nnU-Net')
print('omentum:')
print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(DSC1), np.nanmedian(DSC1)))
print('pelvic/ovarian:')
print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(DSC9), np.nanmedian(DSC9)))

# %% now compute scores for nnU-Net
predp = 'D:\\PhD\\Data\\nnUnet_raw_data_base\\nnUNet_predictions'
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
                          'ApolloTCGA', 'labels')

DSC9 = []
DSC1 = []

for case in tqdm(os.listdir(rawp)):
    if di[case[5:8]]['dataset'] == 'Apollo':
        gt = nib.load(os.path.join(rawp, case)).get_fdata()
        pod = (gt == 9).astype(float)
        om = (gt == 1).astype(float)
        if pod.max() > 0:
            
            pred = nib.load(os.path.join(predp, '120', case)).get_fdata()
            DSC9.append(200 * np.sum(pod*pred) / np.sum(pod+pred))
        
        if om.max() > 0:
            
            pred = nib.load(os.path.join(predp, '121', case)).get_fdata()
            DSC1.append(200 * np.sum(om*pred) / np.sum(om+pred))
print('BARTS')
print('nnU-Net')
print('omentum:')
print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(DSC1), np.nanmedian(DSC1)))
print('pelvic/ovarian:')
print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(DSC9), np.nanmedian(DSC9)))
       
