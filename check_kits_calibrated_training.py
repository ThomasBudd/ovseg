import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

bp = os.path.join(os.environ['OV_DATA_BASE'],
                  'trained_models',
                  'kits21_trn',
                  'disease_3_1')

models = [m for m in os.listdir(bp) if m.startswith('UQ')]

metric = 'dice'

w_list = [float(m[14:]) for m in models]

CV_DSCS = []
TST_DSCS = []

for m in models:
    
    cv_res = pickle.load(open(os.path.join(bp, m, 'validation_CV_results.pkl'), 'rb'))
    
    dsc_tumor = np.nanmean([cv_res[case][f'{metric}_2'] for case in cv_res])
    dsc_cyst = np.nanmean([cv_res[case][f'{metric}_3'] for case in cv_res])
    
    CV_DSCS.append((dsc_tumor, dsc_cyst))
    
    tst_res = pickle.load(open(os.path.join(bp, m, 'ensemble_3_4_5',
                                           'kits21_tst_results.pkl'), 'rb'))
    
    dsc_tumor = np.nanmean([tst_res[case][f'{metric}_2'] for case in tst_res])
    dsc_cyst = np.nanmean([tst_res[case][f'{metric}_3'] for case in tst_res])
    
    TST_DSCS.append((dsc_tumor, dsc_cyst))

CV_DSCS = np.array(CV_DSCS)
TST_DSCS = np.array(TST_DSCS)

plt.close()
plt.subplot(1,2,1)
plt.plot(w_list, CV_DSCS[:, 0], 'bo')
plt.plot(w_list, TST_DSCS[:, 0], 'ro')
plt.xlabel('w')
plt.ylabel(metric)
plt.legend(['Cross-validation', 'Test'])
plt.title('Tumour')
plt.subplot(1,2,2)
plt.plot(w_list, CV_DSCS[:, 1], 'bo')
plt.plot(w_list, TST_DSCS[:, 1], 'ro')
plt.xlabel('w')
plt.legend(['Cross-validation', 'Test'])
plt.title('Cyst')

