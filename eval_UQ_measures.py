import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

with open(os.path.join(os.environ['OV_DATA_BASE'], 'predictions',
                       'OV04', 'pod_om_4fCV','UQ_measures.pkl'), 'rb') as file:
    
    m = pickle.load(file)

def cor(x,y):
    return np.corrcoef(x,y)[0,1]

# %% compute correlation
for cl in [1,9]:
    print(cl)
    
    for key in ['UQ_old', 'UQ_new']:
        print(key)
        gtDSC = np.array(m[cl]['gt'])[:, 0]
        I = np.logical_not(np.isnan(gtDSC))
        gtDSC = gtDSC[I]
        DSChat = np.array(m[cl][key])[I, 0]
        
        gtMV = np.array(m[cl]['gt'])[:, 1]
        MVhat = np.array(m[cl][key])[:, 1]
        UQ = np.array(m[cl][key])[:, 2]
        
        print(f'{cor(gtDSC,DSChat):.3f} {cor(gtMV,MVhat):.3f} {cor(gtMV,UQ):.3f} ')
