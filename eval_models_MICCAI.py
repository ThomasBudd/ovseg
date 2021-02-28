import numpy as np
import pickle
from os import listdir, environ
from os.path import join

mbp = join(environ['OV_DATA_BASE'], 'models_MICCAI', 'OV04')

joined_models = [model for model in listdir(mbp) if model.startswith('joined')]
joined_win_models = sorted([model for model in joined_models if model.find('win') >= 0])
joined_HU_models = sorted([model for model in joined_models if model.find('HU') >= 0])
seq_seg_models = [model for model in listdir(mbp) if model.startswith('segmentation_')]

i1, i2, i3 = 378, 458, 453
# %%
for model in seq_seg_models:
    print(model)
    try:
        res = pickle.load(open(join(mbp, model, 'validation_CV_results.pkl'), 'rb'))
    except FileNotFoundError:
        print('File not found.')
        continue
    keys = [key for key in res if i1 <= int(key[5:8]) <= i2 and int(key[5:8]) != i3]
    dices = [res[key]['dice_1'] for key in keys if res[key]['has_fg_1']]
    print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(dices), np.nanmedian(dices)))

# %%
for model_list in [joined_win_models, joined_HU_models]:
    for model in model_list:
        print(model)
        try:
            res = pickle.load(open(join(mbp, model, 'validation', 'validation_results.pkl'), 'rb'))
        except FileNotFoundError:
            print('File not found.')
            continue
        keys = [key for key in res if i1 <= int(key[5:8]) <= i2 and int(key[5:8]) != i3]
        dices = [res[key]['dice_1'] for key in keys if res[key]['has_fg_1']]
        print('mean: {:.3f}, median: {:.3f}'.format(np.nanmean(dices), np.nanmedian(dices)))