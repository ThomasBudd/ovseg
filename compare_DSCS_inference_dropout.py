import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

bp = os.path.join(os.environ['OV_DATA_BASE'],'trained_models',
                  'kits21_trn', 'disease_3_1')

models = [f'dropout_UNet_{i}' for i in range(4)]
wd_list = np.logspace(-4,-5,4)

mean_scores = []

for i in range(4):
    
    resp = os.path.join(bp, models[i], 'fold_3')
    
    scores = []
    
    for j in range(7):
        
        res = pickle.load(open(os.path.join(resp, f'kits21_tst_{j}_results.pkl'),'rb'))
        
        scores.append(np.mean([res[case]['dice_2'] for case in res]))
    
    plt.plot(7*[i], scores, 'bo')
    mean_scores.append(scores)

plt.plot(np.mean(mean_scores, 1), 'r')
plt.plot(np.mean(mean_scores, 1), 'g')