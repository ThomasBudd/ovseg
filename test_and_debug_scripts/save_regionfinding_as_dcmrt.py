import os
import pickle
from ovseg.utils.io import save_dcmrt_from_data_tpl, read_nii, read_dcms
from skimage.transform import resize

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 'OV04', 'multiclass',
                     'Regionfinding_0.02', 'cross_validation')
di = pickle.load(open(os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'data_info.pkl'),
                      'rb'))
dcmbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04_dcm')

# %%
cases = [case for case in os.listdir(predp) if case.endswith('.nii.gz')]
for case in cases[1:11]:
    info = di[case[:8]]
    patf = os.path.join(dcmbp, info['pat_id'])
    t = info['date']
    
    if len(os.listdir(patf)) != 1:
        scans = [s for s in os.listdir(patf) if s.find(t) >=0]
        
        if len(scans) != 1:
            print('skipping case {}. Found {} matching folders for timepoint {}'.format(case, len(scans), t))
            continue
    else:
        scans = os.listdir(patf)
    
    dcmp = os.path.join(patf, scans[0])
    
    data_tpl = read_dcms(dcmp)
    pred,_,_ = read_nii(os.path.join(predp, case[:8] + '.nii.gz'))
    if pred.shape[0] != data_tpl['image'].shape[0]:
        print('resizing...')
        pred = resize(pred, data_tpl['image'].shape, order=0)
        print('done!')
    data_tpl['pred'] = pred
    
    save_dcmrt_from_data_tpl(data_tpl, os.path.join(predp, info['pat_id']+'_'+t+'_0.02'),
                             'pred')

# %%
import numpy as np
import matplotlib.pyplot as plt
k = 9
pred_cl = pred == k
# contains = np.where(np.sum(data_tpl['label']==k, (1,2)) > 0)[0]
contains = np.where(np.sum(pred_cl, (1,2)) > 0)[0]
np.random.shuffle(contains)
for i, z in enumerate(contains):
    if i > 19:
        continue
    plt.figure()
    im = data_tpl['image'].clip(-150, 250)
    imz = (im[z] - im[z].min()) / (im[z].max() - im[z].min())
    plt.imshow(np.stack([imz, imz, imz+pred_cl[z]],-1)/2)
    plt.contour(data_tpl['label'][z] == k)