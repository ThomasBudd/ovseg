import os
from ovseg.utils.io import read_nii, load_pkl
import numpy as np
import matplotlib.pyplot as plt

data_name = 'OV04'
preprocessed_name = 'pod_om_08_5'
model_name = 'bs4'

imp = os.path.join(os.environ['OV_DATA_BASE'],
                   'raw_data',
                   'OV04',
                   'images')
gtp = os.path.join(os.environ['OV_DATA_BASE'],
                   'raw_data',
                   'OV04',
                   'labels')
predp = os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     data_name,
                     preprocessed_name,
                     model_name)

modelp = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models',
                      data_name, preprocessed_name, model_name)

trn_dscs1_nan = [[] for _ in range(276)]
trn_dscs9_nan = [[] for _ in range(276)]

for f in range(5):
    results = load_pkl(os.path.join(modelp,
                                    'fold_'+str(f),
                                    'training_results.pkl'))
    
    for c in range(276):
        key = 'case_%03d' % c
        if key in results:
            trn_dscs1_nan[c].append(results[key]['dice_1'])
            trn_dscs9_nan[c].append(results[key]['dice_9'])

# %%
trn_dscs1 = []
trn_dscs9 = []
has_1 = []
has_9 = []

for c in range(276):
    if not np.isnan(trn_dscs1_nan[c][0]):
        trn_dscs1.append(trn_dscs1_nan[c])
        has_1.append(c)
    if not np.isnan(trn_dscs9_nan[c][0]):
        trn_dscs9.append(trn_dscs9_nan[c])
        has_9.append(c)

# to np ndarray
trn_dscs1 = np.array(trn_dscs1)
trn_dscs9 = np.array(trn_dscs9)

# %%
# sort
trn_dscs1 = np.sort(trn_dscs1, 1)
trn_dscs9 = np.sort(trn_dscs9, 1)

ind1 = np.argsort(trn_dscs1[:, 0])
ind9 = np.argsort(trn_dscs9[:, 0])

trn_dscs1 = trn_dscs1[ind1]
trn_dscs9 = trn_dscs9[ind9]

plt.subplot(1,2,1)
for i in range(4):
    plt.plot(trn_dscs1[:, i], 'bo')
plt.title('Omentum', fontsize=18)
plt.ylabel('Training DSC', fontsize=18)
plt.ylim([0,100])
plt.subplot(1,2,2)
for i in range(4):
    plt.plot(trn_dscs9[:, i], 'bo')
plt.title('Pelvis/ovaries', fontsize=18)
plt.ylabel('Training DSC', fontsize=18)
plt.ylim([0,100])
# %% get the cases with poor dscs
print('Omentum cases:')
print(ind1[:5])
print('POD cases:')
print(ind9[:5])

colors = ['b', 'r', 'g', 'y']

plotp = os.path.join(os.environ['OV_DATA_BASE'],
                     'plots',
                     data_name,
                     preprocessed_name,
                     model_name,
                     'training_fails')
if not os.path.exists(plotp):
    os.makedirs(plotp)

for C in ind9[:10]:
    
    c = has_9[C]
    print(c)
    
    im = read_nii(os.path.join(imp, 'case_%03d_0000.nii.gz' % c))[0].clip(-150, 250)
    im = (im + 150) / 400
    gt = read_nii(os.path.join(gtp, 'case_%03d.nii.gz' % c))[0]
    lb = (gt == 9).astype(float)
    
    ovlp = np.stack([im + 0.7*lb, im, im], -1) / 1.7
    
    trn_preds = []
    for f in range(5):
        pp = os.path.join(predp, 'training_fold_%d' % f)
        
        if 'case_%03d.nii.gz' % c in os.listdir(pp):
            
            pred = read_nii(os.path.join(pp, 'case_%03d.nii.gz' % c))[0]
            pred = (pred == 9).astype(float)
            trn_preds.append(pred)
            print('{:.3f}'.format(200 * np.sum(lb * pred) / np.sum(lb + pred)))
    
    plotcp = os.path.join(plotp, 'case_%03d_pod' % c)
    if not os.path.exists(plotcp):
        os.mkdir(plotcp)
    
    for z in range(ovlp.shape[0]):
        plt.imshow(ovlp[z])
    
        for col, pred in zip(colors, trn_preds):
            plt.contour(pred[z], linewidths=0.5, colors=col)
        
        plt.axis('off')
        plt.savefig(os.path.join(plotcp, 'slice_%03d.png' % z))
        plt.close()
        
for C in ind1[:10]:
    
    c = has_1[C]
    print(c)
    
    im = read_nii(os.path.join(imp, 'case_%03d_0000.nii.gz' % c))[0].clip(-150, 250)
    im = (im + 150) / 400
    gt = read_nii(os.path.join(gtp, 'case_%03d.nii.gz' % c))[0]
    lb = (gt == 1).astype(float)
    
    ovlp = np.stack([im + 0.7*lb, im, im], -1) / 1.7
    
    trn_preds = []
    for f in range(5):
        pp = os.path.join(predp, 'training_fold_%d' % f)
        
        if 'case_%03d.nii.gz' % c in os.listdir(pp):
            
            pred = read_nii(os.path.join(pp, 'case_%03d.nii.gz' % c))[0]
            pred = (pred == 1).astype(float)
            trn_preds.append(pred)
            print('{:.3f}'.format(200 * np.sum(lb * pred) / np.sum(lb + pred)))
    
    plotcp = os.path.join(plotp, 'case_%03d_om' % c)
    if not os.path.exists(plotcp):
        os.mkdir(plotcp)
    
    for z in range(ovlp.shape[0]):
        plt.imshow(ovlp[z])
    
        for col, pred in zip(colors, trn_preds):
            plt.contour(pred[z], linewidths=0.5, colors=col)
        
        plt.axis('off')
        plt.savefig(os.path.join(plotcp, 'slice_%03d.png' % z))
        plt.close()
