import numpy as np
import pickle
from os import environ, listdir
from os.path import join, exists

modelp = join(environ['OV_DATA_BASE'], 'trained_models', 'OV04')

lucian_ids = [str(i) for i in range(387, 459) if i != 453]
hilal_ids = [str(i) for i in range(459, 529) if i != 525]
# models of interest
joined_ov_pod_models = [m for m in listdir(modelp) if m.startswith('joined_ov_pod')]
models_variance = [m for m in listdir(modelp) if m.startswith('joined_ov_pod') and m[-1].isdigit()]

groups = np.unique([m[:-2] for m in models_variance])

results = {}

for group in groups:
    print(group)
    group_models = [m for m in joined_ov_pod_models if m.startswith(group)]
    group_dices = []
    for model in group_models:
        if not exists(join(modelp, model, 'validation', 'validation_results.pkl')):
            continue
        res = pickle.load(open(join(modelp, model, 'validation', 'validation_results.pkl'), 'rb'))
        keys_lucian = [key for key in res if key[5:8] in lucian_ids]
        keys_hilal = [key for key in res if key[5:8] in hilal_ids]
        dices_lucian = [res[key]['dice_1'] for key in keys_lucian if res[key]['has_fg_1']]
        dices_hilal = [res[key]['dice_1'] for key in keys_hilal if res[key]['has_fg_1']]
        group_dices.append([dices_hilal, dices_lucian])

    if len(group_dices) > 0:
        results[group] = group_dices
        print([np.mean(dices[0]) for dices in group_dices])

pickle.dump(results, open('joint_variance_results.pkl', 'wb'))
