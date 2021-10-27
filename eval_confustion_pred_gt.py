import numpy as np
from os.path import join, exists
from os import environ, makedirs
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import pickle
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from ovseg.data.Dataset import raw_Dataset

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--debug", default=False, action='store_true')
args = parser.parse_args()

lb_classes = [1, 2, 9, 13, 15, 17]
prev_stages = [{'data_name':'OV04',
                'preprocessed_name': 'om_067',
                'model_name': 'larger_res_encoder',
                'lb_classes': [1]},
                {'data_name': 'OV04',
                'preprocessed_name': 'multiclass_1_2_9',
                'model_name': 'U-Net5',
                'lb_classes': [2]},
                {'data_name':'OV04',
                'preprocessed_name': 'pod_067',
                'model_name': 'larger_res_encoder',
                'lb_classes': [9]},
                {'data_name': 'OV04',
                'preprocessed_name': 'multiclass_13_15_17',
                'model_name': 'U-Net5',
                'lb_classes': [13, 15, 17]}]
# prev_stages = [{'data_name':'OV04',
#                 'preprocessed_name': 'pod_om_08_5',
#                 'model_name': 'U-Net4_prg_lrn',
#                 'lb_classes': [1, 9]},
#                 {'data_name': 'OV04',
#                 'preprocessed_name': 'multiclass_1_2_9',
#                 'model_name': 'U-Net5',
#                 'lb_classes': [2]},
#                {'data_name': 'OV04',
#                 'preprocessed_name': 'multiclass_13_15_17',
#                 'model_name': 'U-Net5',
#                 'lb_classes': [13, 15, 17]}]

keys_for_previous_stages = []
for prev_stage in prev_stages:
    for key in ['data_name', 'preprocessed_name', 'model_name']:
        assert key in prev_stage
    key = '_'.join(['prediction',
                    prev_stage['data_name'],
                    prev_stage['preprocessed_name'],
                    prev_stage['model_name']])

    keys_for_previous_stages.append(key)

ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', 'OV04'),
                 prev_stages=prev_stages)

def get_preds(data_tpl):
    preds = []
    confm = np.zeros((6, 6))
    for prev_stage, key in zip(prev_stages, keys_for_previous_stages):
        assert key in data_tpl, 'prediction '+key+' from previous stage missing'
        pred = data_tpl[key]
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        # ensure the array is 3d
        pred = maybe_add_channel_dim(pred)[0]
        
        for cl in prev_stage['lb_classes']:
            preds.append((pred == cl).astype(float))
    
    preds = np.stack(preds)
                
    return preds

def eval_confusion(lbs, preds):
    
    conf = np.zeros((6, 6))
    has_fg = np.zeros((6, 2))

    for i in range(6):
        has_fg[i, 0] = lbs[i].max()
        has_fg[i, 1] = preds[i].max()
        
        for j in range(6):
            conf[i, j] = (lbs[i] * preds[j]).max()
    
    return conf, has_fg
        
# %%
confusion = np.zeros((6, 6))
fgs = np.zeros((6, 2))

if args.debug:
    N = 6
else:
    N = len(ds)

for i in tqdm(range(N)):
    data_tpl = ds[i]
    lb = data_tpl['label']
    lbs = np.stack([(lb == cl).astype(float) for cl in lb_classes])
    preds = get_preds(data_tpl)
    confm, has_fg = eval_confusion(lbs, preds)
    
    confusion += confm
    fgs += has_fg

np.save(join(environ['OV_DATA_BASE'], 'confusion_OV04.npy'), confusion)
np.save(join(environ['OV_DATA_BASE'], 'fgs_OV04.npy'), fgs)

# %%
confusion = np.load(join(environ['OV_DATA_BASE'], 'confusion_OV04.npy'))
fgs = np.load(join(environ['OV_DATA_BASE'], 'fgs_OV04.npy'))

print('number of fg scans:')
print('ground truth')
for i, cl in enumerate(lb_classes):
    print('{}: {:03d}'.format(cl, int(fgs[i, 0])))
print('prediction')
for i, cl in enumerate(lb_classes):
    print('{}: {:03d}'.format(cl, int(fgs[i, 1])))
print()

print('Confusion:')
for i, cl in enumerate(lb_classes):
    conf_str = ' '.join(['{:03d}'.format(int(conf)) for conf in confusion[i]])
    print(str(cl) + ' ' + conf_str)

