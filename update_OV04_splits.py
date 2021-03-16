import pickle
import os

cases_val = ['case_%03d.npy'%i for i in range(459, 530)]

prep = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04')

for pn in os.listdir(prep):
    sp = os.path.join(prep, pn, 'splits.pkl')
    splits = pickle.load(open(sp, 'rb'))
    splits[0]['val'] = cases_val
    pickle.dump(splits, open(sp, 'wb'))
