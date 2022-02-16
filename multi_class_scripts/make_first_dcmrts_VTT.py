import nibabel as nib
from ovseg.utils.io import read_dcms, save_dcmrt_from_data_tpl
from os import listdir, environ
from os.path import join, basename
import numpy as np

predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'bin_seg', 'U-Net5_M_15',
             'BARTS_small_ensemble_0_1_2_3_4')
gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS_dcm')

cases = listdir(gtp)[:10]

automated_labels = np.random.choice(cases, size=5, replace=False)

for i, case in enumerate(cases):

    data_tpl = read_dcms(join(gtp, case))
    pred = nib.load(join(predp, data_tpl['pat_id']+'_'+data_tpl['date']+'.nii.gz')).get_fdata()
    
    if case in automated_labels:
        data_tpl['prediction'] = pred
    else:
        data_tpl['prediction'] = (data_tpl['label'] > 0).astype(float)
    
    
    out_file = join(predp, 'VTT_test_1', basename(data_tpl['raw_label_file']))
    save_dcmrt_from_data_tpl(data_tpl, out_file, key='prediction', names=['1-HGSOC'])
    
np.save(join(predp, 'VTT_test_1', 'automated_labels.npy'), automated_labels)
# %%


automated_labels = np.load(join(predp, 'VTT_test_1', 'automated_labels.npy'))
