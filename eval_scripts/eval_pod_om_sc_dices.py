import os
import nibabel as nib
from tqdm import tqdm
import numpy as np

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 'OV04')
mcp = os.path.join(predp, 'pod_om_10', 'res_encoder', 'cross_validation')
gtp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'labels')

def dice(s1, s2):
    return 200 * np.sum(s1*s2)/np.sum(s1+s2)

mc_dices = []
sc_dices = []
mc_sc_dices = []

for case in tqdm(os.listdir(mcp)):
    mc = (nib.load(os.path.join(mcp, case)).get_fdata() > 0).astype(float)
    gt = nib.load(os.path.join(gtp, case)).get_fdata()
    gt = ((gt == 9) + (gt == 1)).astype(float)
    pod = nib.load(os.path.join(predp, 'pod_067', 'larger_res_encoder', 'cross_validation',
                                case)).get_fdata()
    om = nib.load(os.path.join(predp, 'om_08', 'res_encoder_no_prg_lrn', 'cross_validation',
                               case)).get_fdata()
    sc = ((pod + om) > 0).astype(float)
    mc_dices.append(dice(mc, gt))
    sc_dices.append(dice(sc, gt))
    mc_sc_dices.append(dice(sc, mc))

# %%
print(np.mean(mc_dices))
print(np.mean(sc_dices))
print(np.mean(mc_sc_dices))