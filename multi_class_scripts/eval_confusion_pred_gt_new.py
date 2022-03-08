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

pred_lb_classes = [1, 2, 9, 13, 15, 17]
gt_lb_classes = [1, 2, 3, 4, 5, 6, 7, 9, 13, 14, 15, 16, 17]

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

def get_preds(data_tpl):
    preds = []
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
    
    conf = np.zeros((len(gt_lb_classes), len(pred_lb_classes)))
    has_gt_fg = np.zeros(len(gt_lb_classes))
    has_pred_fg = np.zeros(len(pred_lb_classes))

    for i in range(len(gt_lb_classes)):
        has_gt_fg[i] = lbs[i].max()
    
    for j in range(len(pred_lb_classes)):
        has_pred_fg[j] = preds[j].max()
    
    
    for i in range(len(gt_lb_classes)):
        has_gt_fg[i] = lbs[i].max()
        for j in range(len(pred_lb_classes)):
            conf[i, j] = (lbs[i] * preds[j]).max()
    
    return conf, has_gt_fg, has_pred_fg 
        
# %%

for ds_name in ['BARTS']:
    
    ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', ds_name),
                     prev_stages=prev_stages)
    confusion = np.zeros((len(gt_lb_classes), len(pred_lb_classes)))
    fgs_gt = np.zeros(len(gt_lb_classes))
    fgs_pred = np.zeros(len(pred_lb_classes))
    
    if args.debug:
        N = 6
    else:
        N = len(ds)
    
    for i in tqdm(range(N)):
        data_tpl = ds[i]
        lb = data_tpl['label']
        lbs = np.stack([(lb == cl).astype(float) for cl in gt_lb_classes])
        preds = get_preds(data_tpl)
        confm, has_gt_fg, has_pred_fg = eval_confusion(lbs, preds)
        
        confusion += confm
        fgs_gt += has_gt_fg
        fgs_pred += has_pred_fg
    
    np.save(join(environ['OV_DATA_BASE'], 'confusion_{}.npy'.format(ds_name)), confusion)
    np.save(join(environ['OV_DATA_BASE'], 'fgs_gt_{}.npy'.format(ds_name)), fgs_gt)
    np.save(join(environ['OV_DATA_BASE'], 'fgs_pred_{}.npy'.format(ds_name)), fgs_pred)
    
    # %%

for ds_name in ['BARTS']:
    
    print(ds_name)
    
    confusion = np.load(join(environ['OV_DATA_BASE'], 'confusion_{}.npy'.format(ds_name)))
    fgs_gt = np.load(join(environ['OV_DATA_BASE'], 'fgs_gt_{}.npy'.format(ds_name)))
    fgs_pred = np.load(join(environ['OV_DATA_BASE'], 'fgs_pred_{}.npy'.format(ds_name)))
    
    print('number of fg scans gt:')
    for i, cl in enumerate(gt_lb_classes):
        print('{:02d}: {:03d}'.format(int(cl), int(fgs_gt[i])))
    print()
    
    print('number of fg scans pred:')
    for i, cl in enumerate(pred_lb_classes):
        print('{:02d}: {:03d}'.format(int(cl), int(fgs_pred[i])))
    print()
    
    print('Confusion:')
    for i, cl in enumerate(gt_lb_classes):
        conf_str = ' '.join(['{:03d}'.format(int(conf)) for conf in confusion[i]])
        print('{:2d}'.format(int(cl)) + ' ' + conf_str)

