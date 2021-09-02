from os import listdir, environ
from os.path import join
import numpy as np
import pickle
import matplotlib.pyplot as plt

bp = join(environ['OV_DATA_BASE'], 'trained_models', 'OV04', 'pod_2d')
num_epochs_list = [250, 500, 750, 1000]

# %%
for ne in num_epochs_list:
    modelp = join(bp, '2d_UNet_small_{}'.format(ne))
    vr = pickle.load(open(join(modelp, 'validation_CV_results.pkl'), 'rb'))
    mvd = np.mean([vr[key]['dice_1'] for key in vr])
    # evalr = pickle.load(open(join(modelp, 'ensemble_0_1_2_3_4', 'BARTS_results.pkl'), 'rb'))
    # mevald = np.mean([evalr[key]['dice_1'] for key in evalr])
    mevald = 0
    print('CV: {:.3f}, Eval: {:.3f}'.format(mvd, mevald))

# %%
val_dices = np.zeros((4, 5))

for i, ne in enumerate(num_epochs_list):
    modelp = join(bp, '2d_UNet_small_{}'.format(ne))
    for j in range(5):
        res = pickle.load(open(join(modelp, 'fold_'+str(j), 'BARTS_results.pkl'),'rb'))
        md = np.mean([res[key]['dice_1'] for key in res])
        val_dices[i, j] = md

for j in range(5):
    plt.plot(num_epochs_list, val_dices[:, j])