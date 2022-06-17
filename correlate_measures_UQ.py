import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import pickle

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','OV04','pod_om_4fCV')
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

n_ens = 7
cl = 9

# %%


model_name = 'calibrated_0.00'

for n_ens in [5,7]:
    
    for cl in [1, 9]:

        measures = []
        
        for ds_name in ['ApolloTCGA', 'BARTS']:
            
            scans = [s for s in os.listdir(os.path.join(predp, model_name, ds_name+'_fold_5'))
                     if s.endswith('.nii.gz')]
        
            sleep(0.1)
            print(f'{n_ens}, {cl}')
            for scan in tqdm(scans):
                
                gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
                gt = (gt == cl).astype(float)
                hm = np.zeros_like(gt)
                
                for f in range(5,5+n_ens):
                    
                    pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_fold_{f}', scan)).get_fdata()
                    hm += (pred == cl).astype(float)
                
                hm = hm/n_ens
                pred = (hm > 0.5).astype(float)
                
                # similarity measures
                if gt.max()>0:
                    DSC = 200*np.sum(gt*pred)/np.sum(gt+pred)
                else:
                    DSC = np.nan
                
                MV = np.sum(np.abs(gt-pred))
                
                # UQ measures
                sm = np.sum(hm+pred)
                if sm > 0:
                    DSC_hat = 200 * np.sum(hm*pred)/sm
                else:
                    DSC_hat = 100
                MV_hat = np.sum(np.abs(hm - pred))
                UNC = np.sum(2*hm*(1-hm))
                
                measures.append([DSC, MV, DSC_hat, MV_hat, UNC])
        
        measures = np.array(measures)
        np.save(os.path.join(predp, f'measures_0_0_{n_ens}_{cl}.npy'), measures)
# %%

# for i in range(3):
#     print(f'{m_names[i]}: {np.corrcoef(gtm, uqm[:, i])[0, 1]:.3f}')

# plt.close()
# plt.plot(gtm, uqm[:, 0], 'bo')

# %%
all_models = [m for m in os.listdir(predp) if m.startswith('calibrated')]
all_w = np.array([float(m.split('_')[1]) for m in all_models])


for n_ens in [5,7,9,11,13,17,21,27,35]:
    w_list = np.linspace(-3, 3, n_ens)
    
    for cl in [1, 9]:

        measures = []
        
        for ds_name in ['ApolloTCGA', 'BARTS']:
            
            scans = [s for s in os.listdir(os.path.join(predp, model_name, ds_name+'_fold_5'))
                     if s.endswith('.nii.gz')]
        
            sleep(0.1)
            print(f'{n_ens}, {cl}')
            for scan in tqdm(scans):
                
                gt = nib.load(os.path.join(rawp, ds_name, 'labels', scan)).get_fdata()
                gt = (gt == cl).astype(float)
                hm = np.zeros_like(gt)
                
                for w in w_list:
            
                    w_ind = np.argmin(np.abs(w-all_w))
                    model_name = all_models[w_ind]
                    
                    pred = nib.load(os.path.join(predp, model_name, f'{ds_name}_fold_5', scan)).get_fdata()
                    hm += (pred == cl).astype(float)
                
                hm = hm/n_ens
                pred = (hm > 0.5).astype(float)
                
                # similarity measures
                if gt.max()>0:
                    DSC = 200*np.sum(gt*pred)/np.sum(gt+pred)
                else:
                    DSC = np.nan
                
                MV = np.sum(np.abs(gt-pred))
                
                # UQ measures
                sm = np.sum(hm+pred)
                if sm > 0:
                    DSC_hat = 200 * np.sum(hm*pred)/sm
                else:
                    DSC_hat = 100
                MV_hat = np.sum(np.abs(hm - pred))
                UNC = np.sum(2*hm*(1-hm))
                
                measures.append([DSC, MV, DSC_hat, MV_hat, UNC])
        
        measures = np.array(measures)
        np.save(os.path.join(predp, f'measures_-3_3_{n_ens}_{cl}.npy'), measures)
# %%
cl = 9

print('Uncalibrated')
for n_ens in [5,7,9,11,13]:
    measures = np.load(os.path.join(predp, f'measures_0_0_{n_ens}_{cl}.npy'))
    gtm = measures[:, 0]
    uqm = measures[:, 2:]
    I = np.logical_not(np.isnan(gtm))
    gtm = gtm[I]
    uqm = uqm[I]
    print(f'{n_ens} DSC: {np.corrcoef(gtm, uqm[:, 0])[0, 1]}')
    
print('Calibrated')
for n_ens in [5,7,9,11,13,17,21,27,35]:
    measures = np.load(os.path.join(predp, f'measures_-3_3_{n_ens}_{cl}.npy'))
    gtm = measures[:, 0]
    uqm = measures[:, 2:]
    I = np.logical_not(np.isnan(gtm))
    uqm = uqm[I]
    print(f'{n_ens} DSC: {np.corrcoef(gtm, uqm[:, 0])[0, 1]}')



