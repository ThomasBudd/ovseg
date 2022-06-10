import os
import pickle

mbp = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models',
                   'OV04', 'pod_om_4fCV')

models = [m for m in os.listdir(mbp) if m.startswith('calib')]

for m in models:
    
    cp = pickle.load(open(os.path.join(mbp, m, 'fold_5', 'attribute_checkpoint.pkl')), 'rb')
    
    if cp['epochs_done'] < 1000:
        print(m)

