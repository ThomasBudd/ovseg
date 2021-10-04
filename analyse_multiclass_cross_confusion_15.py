import nibabel as nib
import numpy as np
from os import listdir, environ
from os.path import join
from tqdm import tqdm


# %%
gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')
predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'new_multiclass_v1', 'new_loss',
             'BARTS_ensemble_0_1_2_3_4')


bin_sens = []
sens_15 = []
dsc_15 = []

sens_cases = []
for case in tqdm(listdir(gtp)):
    gt = (nib.load(join(gtp, case)).get_fdata() == 15).astype(float)
    
    if gt.max() > 0:
        pred = nib.load(join(predp, case)).get_fdata()
        bin_pred = (pred > 0).astype(float)
        pred_15 = (pred == 15).astype(float)
        bin_sens.append(100 * np.sum(gt*bin_pred) / np.sum(gt))
        sens_15.append(100 * np.sum(gt*pred_15) / np.sum(gt))
        dsc_15.append(200 * np.sum(gt*pred_15) / np.sum(gt+pred_15))
        if bin_sens[-1] > 0:
            sens_cases.append(case)

bin_sens, sens_15, dsc_15 = np.array(bin_sens), np.array(sens_15), np.array(dsc_15)

for i in range(len(bin_sens)):
    print('{}: {:.2f}, {:.2f}, {:.2f}'.format(i, bin_sens[i], sens_15[i], dsc_15[i]))

# %%

for case in sens_cases:
    gt = (nib.load(join(gtp, case)).get_fdata() == 15).astype(float)
    pred = nib.load(join(predp, case)).get_fdata()
    classes = np.unique(pred)
    classes = [cl for cl in classes if cl > 0]
    cl_sens = []
    for cl in classes:
        p = (pred == cl).astype(float)
        cl_sens.append(100 * np.sum(p*gt)/np.sum(gt))
    print('{}, {}'.format(case, classes[np.argmax(cl_sens)]))
    
    
# %%
gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')
predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'diaph_reg_expert', 'U-Net2',
             'BARTS_fold_0')

fps = []
tns = []
for case in tqdm(listdir(gtp)):
    gt = (nib.load(join(gtp, case)).get_fdata() == 15).astype(float)
    
    if gt.max() == 0:
        pred = nib.load(join(predp, case)).get_fdata()
        if pred.max() > 0:
            fps.append(case)
        else:
            tns.append(case)

n = len(fps) + len(tns)

print('{} out out {} cases had false positives ({:.1f}%)'.format(len(fps), n, 100*len(fps)/n))

# %%
gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')
predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'bin_seg', 'U-Net5_M_15',
             'BARTS_ensemble_0_1_2_3_4')
sens = []
for case in tqdm(listdir(gtp)):
    gt = (nib.load(join(gtp, case)).get_fdata() == 15).astype(float)
    
    if gt.max() > 0:
        pred = nib.load(join(predp, case)).get_fdata()
        sens.append(100 * np.sum(gt * pred) / np.sum(gt))

