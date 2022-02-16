import numpy as np
from ovseg import OV_PREPROCESSED
from os import listdir, environ
from os.path import join
from tqdm import tqdm

prep = join(OV_PREPROCESSED, 'OV04', 'reg_expert_13_15_17')

lb_fol = join(prep, 'labels')
reg_fol = join(prep, 'regions')


sens_list = []
for case in tqdm(listdir(lb_fol)):
    
    lb = np.load(join(lb_fol, case))
    reg = np.load(join(reg_fol, case))
    
    sens = np.zeros(3)
    for i in range(3):
        lbc = (lb == i+1).astype(float)
        regc = (reg == i+1).astype(float)
        
        if lbc.max() > 0:
            sens[i] = 100 * np.sum(lbc * regc) / np.sum(lbc)
        else:
            sens[i] = np.nan
    
    sens_list.append(sens)

sens_list = np.stack(sens_list, 0)

mean_sens = np.nanmean(sens_list, 0)
for i in range(3):
    print('Region {}: sens: {:.2f}'.format(i, mean_sens[i]))
