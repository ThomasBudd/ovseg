import os
import nibabel as nib
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

dp = 'PATH_TO_KITS_DATA_FOLDER' # should be .../kits21/kits21/data
tp = os.path.join(os.environ['OV_DATA_BASE'],
                  'raw_data', 'kits21')

timp = os.path.join(tp, 'images')
tlbp = os.path.join(tp, 'labels')

for p in [timp, tlbp]:
    if not os.path.exists(p):
        os.makedirs(p)

for case in tqdm(os.listdir(dp)):
    
    if os.path.exists(os.path.join(dp, case, 'aggregated_MAJ_seg.nii.gz')):
        shutil.copy(os.path.join(dp, case, 'imaging.nii.gz'), 
                    os.path.join(timp, case+'.nii.gz'))
        shutil.copy(os.path.join(dp, case, 'aggregated_MAJ_seg.nii.gz'),
                    os.path.join(tlbp, case+'.nii.gz'))
    