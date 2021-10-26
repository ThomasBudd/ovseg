from os import rename, environ, listdir, makedirs
from os.path import join, exists
import nibabel as nib
from tqdm import tqdm
from time import sleep
import numpy as np

rbp = join(environ['OV_DATA_BASE'], 'raw_data', 'Lits')

folders = ['Training Batch 1', 'Training Batch 2']

lbp = join(rbp, 'labels')
imp = join(rbp, 'images')
for f in [lbp, imp]:
    if not exists(f):
        makedirs(f)

for folder in folders:
    
    dp = join(rbp, folder)
    print(folder)
    sleep(0.1)
    for file in tqdm(listdir(dp)):
        
        i = int(file.split('-')[1][:-4])
        
        img = nib.load(join(dp, file))
        # print(file)
        if file.startswith('volume'):
            
            nib.save(img, join(imp, 'case_{:03d}_0000.nii.gz'.format(i)))
            # rename(join(dp, file), join(imp, 'case_{:03d}_0000.nii.gz'.format(i)))
        elif file.startswith('segmentation'):
            nib.save(img, join(lbp, 'case_{:03d}.nii.gz'.format(i)))
        
        else:
            raise ValueError('Got file {}'.format(file))

# %%
sp_list = []
shape_list = []
for case in tqdm(listdir(lbp)):
    img = nib.load(join(lbp, case))
    sp_list.append(img.header['pixdim'][1:4])
    shape_list.append(img.shape)

sp_list = np.array(sp_list)

print(np.mean(sp_list, 0))
print(np.median(np.array(shape_list), 0))