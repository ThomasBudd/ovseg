import os
import shutil
from tqdm import tqdm
import numpy as np
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

ds_names = ['kits21', 'Lits19']

for ds_name in ds_names:
    
    print(ds_name)
    
    scans = os.listdir(os.path.join(rawp, ds_name, 'labels'))
    np.random.shuffle(scans)

    n_trn_scans = len(scans) // 5 * 3
    trn_scans = scans[:n_trn_scans]
    tst_scans = scans[n_trn_scans:]
    
    for ext, scans in zip(['_trn', 'tst'], [trn_scans, tst_scans]):
        for subf in ['images', 'labels']:
            
            p = os.path.join(rawp, f'{ds_name}_{ext}', subf)
            if not os.path.exists(p):
                os.makedirs(p)
    
        for scan in tqdm(scans):
            
            shutil.copy(os.path.join(rawp, ds_name, 'labels', scan),
                        os.path.join(rawp, f'{ds_name}_{ext}', 'labels', scan))
            
            im_name = [s for s in os.listdir(os.path.join(rawp, ds_name, 'images'))
                       if s.startswith(scan.split('.')[0])][0]
        
            shutil.copy(os.path.join(rawp, ds_name, 'images', im_name),
                        os.path.join(rawp, f'{ds_name}_{ext}', 'images', im_name))
        
        