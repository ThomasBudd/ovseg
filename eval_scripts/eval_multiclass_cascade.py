import numpy as np
import nibabel as nib
import os
from tqdm import tqdm

gtp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 'OV04')
mcp = os.path.join(predp, 'multiclass_cascade_08', 'res_encoder_no_cascade',
                   'BARTS_ensemble_0_1_2_3_4')
csp = os.path.join(predp, 'multiclass_cascade_08', 'res_encoder', 'BARTS_ensemble_0_1_2_3_4')

smps = [os.path.join(predp, 'pod_067', 'larger_res_encoder', 'BARTS_ensemble_0_1_2_3_4')]
for pn in ['lesions_center', 'lesions_lymphnodes', 'lesions_upper', 'om_08']:
    smps.append(os.path.join(predp, pn, 'res_encoder_no_prg_lrn', 'BARTS_ensemble_0_1_2_3_4'))

def DSC(s1, s2):
    return 200 * np.sum(s1 * s2) / np.sum(s1 + s2)

# %% compute the DSC between binary and cascade segmetnation
dsc_bin_cas = []
dsc_gt_sm = []
dsc_gt_cas = []
dsc_gt_mc = []
dsc_cas_mc = []


for case in tqdm(os.listdir(csp)):
    casprd = (nib.load(os.path.join(csp, case)).get_fdata() > 0).astype(float)
    smprd = (np.sum([nib.load(os.path.join(smp, case)).get_fdata() for smp in smps],0) > 0).astype(float)
    mcprd = (nib.load(os.path.join(mcp, case)).get_fdata() > 0).astype(float)
    gt = nib.load(os.path.join(gtp, case)).get_fdata()
    bin_gt = np.zeros_like(gt)
    for c in [1, 2, 3, 4, 5, 6, 7, 9, 13, 14, 15]:
        bin_gt[gt == c] = 1
    dsc_gt_cas.append(DSC(casprd, bin_gt))
    dsc_gt_mc.append(DSC(mcprd, bin_gt))
    dsc_cas_mc.append(DSC(casprd, mcprd))
    dsc_bin_cas.append(DSC(casprd, smprd))
    dsc_gt_sm.append(DSC(bin_gt, smprd))

print('cascade to single models', np.mean(dsc_bin_cas))
print('cascade to multiclass', np.mean(dsc_cas_mc))
print('ground truht to cascade', np.mean(dsc_gt_cas))
print('ground truht to multiclass',np.mean(dsc_gt_mc))
print('ground truht to single class',np.mean(dsc_gt_sm))
# %%