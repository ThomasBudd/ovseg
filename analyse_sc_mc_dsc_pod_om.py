import nibabel as nib
import os
import numpy as np
from tqdm import tqdm

gtp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')
predbp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 'OV04')

pns = ['pod_067', 'om_08', 'multiclass']
mns = ['larger_res_encoder', 'res_encoder_no_prg_lrn', 'Regionfinding_0.02']
dsn = ['BARTS_ensemble_0_1_2_3_4', 'BARTS_ensemble_0_1_2_3_4', 'BARTS_fold_0']

ps = [os.path.join(predbp, pn, mn, ds) for pn, mn, ds in zip(pns, mns, dsn)]

def DSC(s1, s2):
    if s1.max() > 0:
        return 200 * np.sum(s1*s2) /(np.sum(s1) + np.sum(s2))
    else:
        return np.nan

# %%
om_dices_sc = []
pod_dices_sc = []
om_dices_mc = []
pod_dices_mc = []

for case in tqdm(os.listdir(gtp)):
    gt = nib.load(os.path.join(gtp, case)).get_fdata()
    om_gt = (gt == 1).astype(float)
    pod_gt = (gt == 9).astype(float)
    pod_sc, om_sc, reg = [nib.load(os.path.join(p, case)).get_fdata() for p in ps]
    om_dices_sc.append(DSC(om_gt, om_sc))
    pod_dices_sc.append(DSC(pod_gt, pod_sc))
    pod_om = ((pod_sc + om_sc) > 0).astype(float)
    pod_mc = ((pod_om * reg) == 9).astype(float)
    om_mc = ((pod_om * reg) == 1).astype(float)
    om_dices_mc.append(DSC(om_gt, om_mc))
    pod_dices_mc.append(DSC(pod_gt, pod_mc))

print('om dices sc: {:.2f}'.format(np.nanmean(om_dices_sc)))
print('om dices mc: {:.2f}'.format(np.nanmean(om_dices_mc)))
print('pod dices sc: {:.2f}'.format(np.nanmean(pod_dices_sc)))
print('pod dices mc: {:.2f}'.format(np.nanmean(pod_dices_mc)))
