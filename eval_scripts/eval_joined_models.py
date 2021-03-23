import numpy as np
import pickle
from os import listdir, environ
from os.path import join, exists

tmp = join(environ['OV_DATA_BASE'], 'trained_models', 'OV04')

models = [model for model in listdir(tmp) if model.startswith('segmentation_on')]

lucian_ids = [str(i) for i in range(387, 459) if i != 453]
hilal_ids = [str(i) for i in range(459, 529) if i != 525]

for model in models:
    if not exists(join(tmp, model, 'validation_CV_results.pkl')):
        continue
    print(model)
    res = pickle.load(open(join(tmp, model, 'validation_CV_results.pkl'), 'rb'))
    keys_lucian = [key for key in res if key[5:8] in lucian_ids]
    keys_hilal = [key for key in res if key[5:8] in hilal_ids]
    dices_lucian = [res[key]['dice_1'] for key in keys_lucian if res[key]['has_fg_1']]
    dices_hilal = [res[key]['dice_1'] for key in keys_lucian if res[key]['has_fg_1']]
    print('mean hilal: {:.3f}, lucian: {:.3f}'.format(np.nanmean(dices_hilal),
                                                      np.nanmedian(dices_lucian)))


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
    keys_lucian = [key for key in res if key[5:8] in lucian_ids]
    keys_hilal = [key for key in res if key[5:8] in hilal_ids]
    dices_lucian = [res[key]['dice_1'] for key in keys_lucian if res[key]['has_fg_1']]
    dices_hilal = [res[key]['dice_1'] for key in keys_lucian if res[key]['has_fg_1']]
    print('mean hilal: {:.3f}, lucian: {:.3f}'.format(np.nanmean(dices_hilal),
                                                      np.nanmedian(dices_lucian)))