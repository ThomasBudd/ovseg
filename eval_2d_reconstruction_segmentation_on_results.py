import pickle
from os import listdir, environ
from os.path import join
import numpy as np

tmp = join(environ['OV_DATA_BASE'], 'trained_models', 'OV04')

ids = [i for i in range(387, 387+71) if i != 453]

cases = ['case_%03d'%i for i in ids]
# cases = ['case_%03d'%i for i in range(276, 276+71)]


# %% first check the reconstructions on full HU
print('PSNR on full HU range: ')
models = ['recon_fbp_convs_full_HU', 'recon_LPD_full_HU']
for model in models:
    key = 'PSNR'
    val_results = pickle.load(open(join(tmp, model, 'validation_CV_results.pkl'), 'rb'))
    stats = []
    for case in cases:
        case_key = [key for key in val_results if key.startswith(case)][0]
        stats.append(val_results[case_key][key])
    print('{}: PSNR: {:.3f}'.format(model, np.nanmean(stats)))


# %% first check the reconstructions
print()
print('PSNR in [-50, 350] HU')
models = ['recon_fbp_convs_full_HU', 'recon_LPD_full_HU', 'reconstruction_network_fbp_convs', 'LPD']
for model in models:
    key = 'PSNR_win' if model.find('HU') >= 0 else 'PSNR'
    val_results = pickle.load(open(join(tmp, model, 'validation_CV_results.pkl'), 'rb'))
    stats = []
    for case in cases:
        case_key = [key for key in val_results if key.startswith(case)]
        if len(case_key) == 0:
            print(case + ' not found')
            continue
        case_key = case_key[0]
        stats.append(val_results[case_key][key])
    print('{}: PSNR: {:.3f}'.format(model, np.nanmean(stats)))

# %% now the corresponding segmentation results
print()
print('DSC for models trainded on the corresponding learned reconstructions')
models = ['segmentation_on_recon_fbp_convs_full_HU',
          'segmentation_on_recon_LPD_full_HU',
          'segmentation_on_reconstruction_network_fbp_convs',
          'segmentation_on_Siemens_recons']
for model in models:
    key = 'dice_1'
    val_results = pickle.load(open(join(tmp, model, 'validation_CV_results.pkl'), 'rb'))
    stats = []
    for case in cases:
        case_key = [key for key in val_results if key.startswith(case)][0]
        stats.append(val_results[case_key][key])
    print('{}: DSC: {:.3f}'.format(model, np.nanmean(stats)))

# %% now the corresponding segmentation results
print()
print('DSC of model trained on siemens scans on leanred reconstructions')
models = ['_recon_fbp_convs_full_HU',
          '_recon_LPD_full_HU',
          '_reconstruction_network_fbp_convs',
          '']

for model in models:
    key = 'dice_1'
    val_results = pickle.load(open(join(tmp, 'segmentation_on_Siemens_recons', 'validation{}_CV_results.pkl'.format(model)),
                                   'rb'))
    stats = []
    for case in cases:
        case_key = [key for key in val_results if key.startswith(case)][0]
        stats.append(val_results[case_key][key])
    print('{}: DSC: {:.3f}'.format(model, np.nanmean(stats)))

