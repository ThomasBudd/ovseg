import numpy as np
import pickle
from os import listdir, environ
from os.path import join

tmp = join(environ['OV_DATA_BASE'], 'trained_models', 'OV04')

models = [model for model in listdir(tmp) if model.startswith('segmentation_on')]

for model in models:
    print(model)
    res = pickle.load(open(join(tmp, model, 'validation_CV_results.pkl'), 'rb'))
    dices = [res[key]['dice_1'] for key in res if res[key]['has_fg_1']]
    print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(dices), np.nanmedian(dices)))


models = [model for model in listdir(tmp) if model.startswith('joined_')]

for model in models:
    print(model)
    try:
        res = pickle.load(open(join(tmp, model, 'validation', 'validation_results.pkl'), 'rb'))
    except FileNotFoundError:
        print('File not found.')
        continue
    dices = [res[key]['dice_1'] for key in res if res[key]['has_fg_1']]
    print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(dices), np.nanmedian(dices)))
