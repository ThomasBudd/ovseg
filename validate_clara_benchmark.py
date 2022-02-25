import numpy as np
import os
import nibabel as nib
from tqdm import tqdm
from ovseg.data.Dataset import raw_Dataset

p = os.path.join(os.environ['OV_DATA_BASE'], 'predictions',
                 'clara_benchmark')

npp = os.path.join(p, 'pred_numpy')
models = ['abdominal_lesions', 'lymph_nodes', 'pod_om']
exts = ['abd', 'lymph', 'pod_om']

#%%

dscs = {key: [] for key in models}
n_diff = {key: [] for key in models}

def bin_dsc(p1, p2):
    p1, p2 = (p1 > 0).astype(float), (p2 > 0).astype(float)
    return 200 * np.sum(p1*p2)/np.sum(p1+p2)

for model, ext in zip(models, exts):
    
    nibp = os.path.join(p, model)
    
    for case in tqdm(os.listdir(nibp)):
        
        nib_pred = nib.load(os.path.join(nibp, case)).get_fdata()
        scan_id = case.split('_')[0]
        npy_pred = np.load(os.path.join(npp, f'pred_{scan_id}_{ext}.npy'))
        
        # some magic hacks...
        npy_pred = np.rot90(npy_pred[::-1, :, ::-1], -1)
        
        n_diff[model].append(np.sum(nib_pred!=npy_pred))
        dscs[model].append(bin_dsc(nib_pred, npy_pred))

# %%
pod_dscs = []
om_dscs = []

model, ext = models[2], exts[2]
nibp = os.path.join(p, model)

ds = raw_Dataset('TCGA_clara_test', create_missing_labels_as_zero=True)

for data_tpl in tqdm(ds):
    
    scan_id = data_tpl['pat_id']
    case = data_tpl['scan']+'.nii.gz'
    nib_pred = nib.load(os.path.join(nibp, case)).get_fdata()
    scan_id = case.split('_')[0]
    npy_pred = np.load(os.path.join(npp, f'pred_{scan_id}_{ext}.npy'))
    
    # some magic hacks...
    npy_pred = np.rot90(npy_pred[::-1, :, ::-1], -1)
    
    gt = np.moveaxis(data_tpl['label'],0, -1)

    pod = gt == 9
    om = gt == 1
    if pod.max() > 0:
        pod_dscs.append((bin_dsc(pod, nib_pred==9), bin_dsc(pod, npy_pred==9)))
    if om.max() > 0:
        om_dscs.append((bin_dsc(om, nib_pred==1), bin_dsc(om, npy_pred==1)))

print(np.mean(pod_dscs, 0))
print(np.mean(om_dscs, 0))
