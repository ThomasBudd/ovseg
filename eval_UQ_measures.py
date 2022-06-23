import numpy as np
import os
import matplotlib.pyplot as plt
import pickle


predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions','OV04','pod_om_4fCV')

with open(os.path.join(predp,'UQ_measures.pkl'), 'rb') as file:
    
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

# %%
fs = 16
plt.close()
for i, cl in enumerate([1,9]):
    plt.subplot(1,2,i+1)
    for color, key in zip(['b','r'],
                                 ['UQ_old', 'UQ_new']):
        gtDSC = np.array(m[cl]['gt'])[:, 0]
        I = np.logical_not(np.isnan(gtDSC))
        gtDSC = gtDSC[I]
        DSChat = np.array(m[cl][key])[I, 0]
        plt.plot(gtDSC, DSChat, color+'o')
    plt.title('Omentum' if i == 0 else 'Pelvic/Ovarian', fontsize=fs)
        
    plt.plot([0, 100], [0, 100], 'k')
    plt.xlabel('True DSC', fontsize=fs)
    plt.ylabel('Predicted DSC', fontsize=fs)
plt.legend(['Uncalibrated ensemble', 'Calibrated ensemble', 'Identity'],
           loc=4, fontsize=fs)


# %%
pms = pickle.load(open(os.path.join(predp,'p_vs_p_measures.pkl'), 'rb') )


P = np.load(os.path.join(predp, 'P_cross_validation.npy'))

fs = 16
plt.close()
for i, cl in enumerate([1,9]):
    plt.subplot(1,2,i+1)
    px_old = np.arange(1,8)/7
    py_old = pms[cl]['k_old']/pms[cl]['n_old']
    plt.plot(px_old, py_old, 'bo')
    
    py_new = pms[cl]['k_new']/pms[cl]['n_new']
    
    plt.plot(P[:, i], py_new, 'ro')
    plt.plot(P[:, i], P[:,i], 'go')
    plt.title('Omentum' if i == 0 else 'Pelvic/Ovarian', fontsize=fs)
        
    plt.plot([0, 1], [0, 1], 'k')
    plt.xlabel('Predicted probability', fontsize=fs)
    plt.ylabel('Relative frequency of foreground', fontsize=fs)
plt.legend(['Uncalibrated ensemble (test)',
            'Calibrated ensemble (test)',
            'Calibrated ensemble (CV)',
            'Identity'],
           loc=4, fontsize=fs)

