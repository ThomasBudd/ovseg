import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


measures_kits = pickle.load(open(os.path.join(os.environ['OV_DATA_BASE'],
                                              'predictions',
                                              'kits21_trn',
                                              'disease_3_1',
                                              'all_UQ_measures.pkl'),'rb'))


# %% p vs p plot
plt.close()
plt.plot(np.arange(1,8)/7, measures_kits[2]['P_gt_drop'][2:], 'bo')
plt.plot(np.arange(1,8)/7, measures_kits[2]['P_gt_old'][2:], 'ro')
plt.plot(measures_kits[2]['P'][1:8],
         measures_kits[2]['P_gt_new'][1:8], 'go')
plt.plot([0, 1], [0, 1], 'k')

plt.legend(['Dropout', 'Uncalibrated ensemble', 'Calibrated ensemble', 'Identity'])


# %%

plt.figure()
plt.subplot(2, 3, 3)
for ext, c in zip(['drop', 'old', 'new'], ['b', 'r', 'g']):
    plt.plot(measures_kits[2][f'DSCs_{ext}'],
             measures_kits[2]['gt'],
             f'{c}o')
plt.plot([0, 100], [0, 100], 'k')

plt.legend(['Dropout', 'Uncalibrated ensemble', 'Calibrated ensemble', 'Identity'])
plt.xlim([50, 100])
plt.ylim([50, 100])

plt.subplot(2,3,6)
plt.plot(measures_kits[2]['P'][1:8],
         measures_kits[2]['P_pred_new'][1:8], 'mo')
plt.plot(measures_kits[2]['P'][1:8],
         measures_kits[2]['P_gt_new'][1:8], 'go')
plt.plot([0, 1], [0, 1], 'k')

