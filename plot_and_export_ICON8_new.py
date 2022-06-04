import os
import pickle
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from ovseg.data.Dataset import raw_Dataset

predbp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions',
                   'ApolloTCGA_BARTS_OV04','pod_om')

# %% analyse the CV performance
mbp = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models',
                   'ApolloTCGA_BARTS_OV04','pod_om')

all_models = [m for m in os.listdir(mbp) if m.startswith('cali')]

w1_list, w9_list = [], []
dsc1_list, dsc9_list = [], []

for model in all_models:
    params = pickle.load(open(os.path.join(mbp, model, 'model_parameters.pkl'), 'rb'))
    res = pickle.load(open(os.path.join(mbp, model, 'validation_CV_results.pkl'), 'rb'))
    w1_list.append(params['training']['loss_params']['loss_kwargs'][0]['w_list'][0])
    w9_list.append(params['training']['loss_params']['loss_kwargs'][0]['w_list'][1])
    dsc1_list.append(np.nanmean([res[scan]['dice_1'] for scan in res]))
    dsc9_list.append(np.nanmean([res[scan]['dice_9'] for scan in res]))

w_dsc1_sorted = np.array([(w, dsc) for w, dsc in sorted(zip(w1_list, dsc1_list))])
w_dsc9_sorted = np.array([(w, dsc) for w, dsc in sorted(zip(w9_list, dsc9_list))])

plt.subplot(1,2,1)
plt.plot(w_dsc1_sorted[:, 0], w_dsc1_sorted[:, 1])
plt.subplot(1,2,2)
plt.plot(w_dsc9_sorted[:, 0], w_dsc9_sorted[:, 1])

# compute weights with DSC better than best - 5
w1 = w_dsc1_sorted[w_dsc1_sorted[:, 1] >= w_dsc1_sorted[:, 1].max()-5, 0]
w9 = w_dsc9_sorted[w_dsc9_sorted[:, 1] >= w_dsc9_sorted[:, 1].max()-5, 0]

# we throw the last w from 9 out for symmetry
w9 = w9[:len(w1)]

models = [m for m in all_models if float(m.split('_')[1]) in w1 or float(m.split('_')[2]) in w9]

# %% plot heatmaps
ds = raw_Dataset('ICON8')
data_tpl = ds[0]
m = models[0]
predp = os.path.join(predbp, m, 'ICON8')
