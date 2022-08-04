import numpy as np
import pickle
import os

bp = os.path.join(os.environ['OV_DATA_BASE'],
                  'trained_models',
                  'OV04',
                  'pod_om_4fCV')

models = [f'dropout_UNet_{i}' for i in range(4)]

mean_scores = np.zeros((4,4))

for i, model in enumerate(models):
    
    res_B_list = [pickle.load(open(os.path.join(bp, model, 'fold_5',
                                                f'BARTS_{j}_results.pkl'), 'rb')) for j in range(8)]
    res_A_list = [pickle.load(open(os.path.join(bp, model, 'fold_5',
                                                f'ApolloTCGA_{j}_results.pkl'), 'rb')) for j in range(8)]
    
    mean_scores[i, 0] = np.mean([np.nanmean([res[case]['dice_1'] for case in res]) for res in res_B_list])
    mean_scores[i, 2] = np.mean([np.nanmean([res[case]['dice_9'] for case in res]) for res in res_B_list])
    mean_scores[i, 1] = np.mean([np.nanmean([res[case]['dice_1'] for case in res]) for res in res_A_list])
    mean_scores[i, 3] = np.mean([np.nanmean([res[case]['dice_9'] for case in res]) for res in res_A_list])

print(mean_scores)