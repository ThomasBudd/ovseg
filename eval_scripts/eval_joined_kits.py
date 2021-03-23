import numpy as np
import pickle
from os import listdir, environ
from os.path import join, exists

tmp = join(environ['OV_DATA_BASE'], 'trained_models', 'kits19')

models = [model for model in listdir(tmp) if model.startswith('segmentation_on')]

for model in models:
    if not exists(join(tmp, model, 'validation_CV_results.pkl')):
        continue
    print(model)
    res = pickle.load(open(join(tmp, model, 'validation_CV_results.pkl'), 'rb'))
    dices_1 = [res[key]['dice_1'] for key in res if res[key]['has_fg_1']]
    dices_2 = [res[key]['dice_2'] for key in res if res[key]['has_fg_2']]
    print('kidney: {:.3f}, tumor: {:.3f}'.format(np.nanmean(dices_1),
                                                 np.nanmedian(dices_2)))


models = [model for model in listdir(tmp) if model.startswith('joined_')]
models_win = sorted([model for model in models if model.endswith('win')])
models_HU = sorted([model for model in models if model.endswith('HU')])
models = models_win + models_HU

for model in models:
    valp = join(tmp, model, 'validation', 'validation_results.pkl')
    if not exists(valp):
        continue
    print(model)
    res = pickle.load(open(valp, 'rb'))
    dices_1 = [res[key]['dice_1'] for key in res if res[key]['has_fg_1']]
    dices_2 = [res[key]['dice_2'] for key in res if res[key]['has_fg_2']]
    print('kidney: {:.3f}, tumor: {:.3f}'.format(np.nanmean(dices_1),
                                                 np.nanmedian(dices_2)))