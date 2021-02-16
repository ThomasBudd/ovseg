import numpy as np
import pickle
from os import listdir, environ
from os.path import join

tmp = join(environ['OV_DATA_BASE'], 'trained_models', 'OV04')

models = [model for model in listdir(tmp) if model.startswith('segmentation_on')]

i1, i2 = 378, 448

for model in models:
    print(model)
    res = pickle.load(open(join(tmp, model, 'validation_CV_results.pkl'), 'rb'))
    keys = [key for key in res if i1 <= int(key[5:8]) <= i2]
    dices = [res[key]['dice_1'] for key in keys if res[key]['has_fg_1']]
    print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(dices), np.nanmedian(dices)))


models = [model for model in listdir(tmp) if model.startswith('joined_')]

i1, i2 = 378, 448

for model in models:
    print(model)
    try:
        res = pickle.load(open(join(tmp, model, 'validation', 'validation_results.pkl'), 'rb'))
    except FileNotFoundError:
        print('File not found.')
        continue
    keys = [key for key in res if i1 <= int(key[5:8]) <= i2]
    dices = [res[key]['dice_1'] for key in keys if res[key]['has_fg_1']]
    print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(dices), np.nanmedian(dices)))
