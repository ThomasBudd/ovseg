import numpy as np
import nibabel as nib
from ovseg.utils.io import read_dcms, save_dcmrt_from_data_tpl
from ovseg.utils.label_utils import reduce_classes
from os import environ, listdir, mkdir
from os.path import join, basename, exists
from tqdm import tqdm
import pickle
from skimage.transform import resize
# %%
predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'new_multiclass_v1', 'new_loss',
             'BARTS_ensemble_0_1_2_3_4')
gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')

lb_classes = [1, 2, 9]
dscs = np.zeros(3)
fps = np.zeros(3)
has_fgs = np.zeros(3)

nii_files = [nii_file for nii_file in listdir(predp) if nii_file.endswith('.nii.gz')]
for case in tqdm(nii_files):
    
    gt = nib.load(join(gtp, case)).get_fdata()
    pred = nib.load(join(predp, case)).get_fdata()

    for i, cl in enumerate(lb_classes):
        l1 = (gt == cl).astype(float)
        l2 = (pred == cl).astype(float)
        
        has_fg = l1.max()
        has_fgs[i] += has_fg
        
        if has_fg:
            dscs[i] += 200 * np.sum(l1*l2)/np.sum(l1 + l2)
        else:
            fps[i] += l2.max()
    

dscs = dscs/has_fgs
n = len(listdir(predp))
fps = 100 * fps / (n - has_fgs)

for i in range(3):
    print('{}: DSC: {:.2f}, fps: {:.2f}'.format(lb_classes[i],
                                                dscs[i],
                                                fps[i]))
    
# %% now export to dcm rt
data_info = pickle.load(open(join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'data_info.pkl'), 'rb'))

gtp_dcm = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS_dcm')
nii_files = [nii_file for nii_file in listdir(predp) if nii_file.endswith('.nii.gz')]

dcmrtp = join(predp, 'dcmrt_1_2_9')

if not exists(dcmrtp):
    mkdir(dcmrtp)

for case in tqdm(nii_files):
    pred = nib.load(join(predp, case)).get_fdata()
    if pred.max() == 0:
        continue
    
    pred = reduce_classes(pred, [1, 2, 9])
    pred = np.moveaxis(pred, -1, 0)
    info = data_info[case[5:8]]
    data_tpl = read_dcms(join(gtp_dcm, info['scan']))
    
    if data_tpl['label'].shape[0] != pred.shape[0]:
        pred = resize(pred, data_tpl['label'].shape, order=0)
    
    data_tpl['prediction'] = pred
    out_file = join(dcmrtp, basename(data_tpl['raw_label_file']))
    save_dcmrt_from_data_tpl(data_tpl, out_file, key='prediction', names=['1', '2', '9'])
    
    
    