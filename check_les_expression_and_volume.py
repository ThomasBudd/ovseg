import numpy as np
import nibabel as nib
from os import listdir, environ
from os.path import join
from tqdm import tqdm
from time import sleep

measures = {}
datasets = ['OV04', 'BARTS', 'ApolloTCGA']

# %%
for ds in datasets:
    print()
    print(ds)
    print()
    lbp = join(environ['OV_DATA_BASE'], 'raw_data', ds, 'labels')
    
    vols = np.zeros(18)
    has_fgs = np.zeros(18)
    
    sleep(0.5)
    for scan in tqdm(listdir(lbp)):
        img = nib.load(join(lbp, scan))
        fac = np.prod(img.header['pixdim'][1:4])
        lb = img.get_fdata()
        
        for c in range(1, 19):
            ROI = (lb == c).astype(float)
            vols[c-1] += np.sum(ROI) * fac
            has_fgs[c-1] += ROI.max()
        
    measures[ds] = {'volumes': vols, 'has_fgs': has_fgs}

# %%
for ds in datasets:
    print()
    print(ds)
    print()
    
    tv = np.sum(measures[ds]['volumes'])/1000
    
    n = len(listdir(join(environ['OV_DATA_BASE'], 'raw_data', ds, 'labels')))
    
    for c in range(18):
        vol = measures[ds]['volumes'][c]/1000
        fg_scans = measures[ds]['has_fgs'][c]
        print('{}: vol = {:.2f}, ({:.2f}%), {:.2f}%'.format(c+1,
                                                            vol/fg_scans if fg_scans > 0 else -1,
                                                            100 * vol/tv,
                                                            100 * fg_scans/n))
        

# %%

for c in range(18):
    s = '{}'.format(c+1).ljust(3)+'|| '
    
    
    for ds in datasets:
        n = len(listdir(join(environ['OV_DATA_BASE'], 'raw_data', ds, 'labels')))
        fg_scans = measures[ds]['has_fgs'][c]
        
        s+= '{:2.1f} '.format(100 * fg_scans/n).ljust(5)
    
    s = s.ljust(23)
    
    s += '|| '
    for ds in datasets:
        vol = measures[ds]['volumes'][c]/1000
        n = len(listdir(join(environ['OV_DATA_BASE'], 'raw_data', ds, 'labels')))
        fg_scans = measures[ds]['has_fgs'][c]
        
        s += '{:3.1f} '.format(vol/fg_scans if fg_scans > 0 else 0).ljust(6)
    
    s = s.ljust(47)
    
    s += '||'
    for ds in datasets:
        tv = np.sum(measures[ds]['volumes'])/1000
        vol = measures[ds]['volumes'][c]/1000
        
        s += '{:2.1f} '.format(100 * vol/tv).ljust(5)
    
    print(s)

