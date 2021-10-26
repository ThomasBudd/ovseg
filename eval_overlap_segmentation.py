import numpy as np
from os.path import join, exists
from os import environ, makedirs
from ovseg.data.Dataset import raw_Dataset
import torch
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--debug", default=False, action='store_true')
args = parser.parse_args()

colors_list = ['red', 'green', 'blue', 'yellow', 'magenta', 'hotpink']

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

keys_for_previous_stages = []
for prev_stage in prev_stages:
    for key in ['data_name', 'preprocessed_name', 'model_name']:
        assert key in prev_stage
    key = '_'.join(['prediction',
                    prev_stage['data_name'],
                    prev_stage['preprocessed_name'],
                    prev_stage['model_name']])

    keys_for_previous_stages.append(key)

ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS'),
                 prev_stages=prev_stages)

def get_pred(data_tpl):
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
    ovlp_list = []
    for i in range(6):
        if preds[i].max() > 0:
            confm[i, i] = 1
        for j in range(i+1, 6):
            s = preds[i] + preds[j]
            if s.max() > 1:
                confm[i, j] = 1 
                contains = np.where(np.sum(s, (1,2)))[0]
                for z in contains:
                    ovlp_list.append((z, i, j))
                
    return preds, confm, ovlp_list

# %%
confusion = np.zeros((6, 6))
plotp = join(environ['OV_DATA_BASE'], 'plots', 'OV04', 'ovlp_predictions')
if not exists(plotp):
    makedirs(plotp)

if args.debug:
    N = 6
else:
    N = len(ds)

for i in tqdm(N):
    data_tpl = ds[i]
    lb = data_tpl['label']
    im = data_tpl['image']
    scan = data_tpl['scan']
    preds, confm, ovlp_list = get_pred(data_tpl)
    
    confusion += confm
    
    if len(ovlp_list) > 0:
        lbs = np.stack([(lb == cl).astype(float) for cl in lb_classes])
    
    for z, i, j in ovlp_list:
        plt.subplot(1, 2, 1)
        plt.imshow(im[z].clip(-150, 250), cmap='bone')
        for k, col in enumerate(colors_list):
            if lbs[k, z].max() > 0:
                plt.contour(lbs[k, z], colors=col)
        
        plt.subplot(1, 2, 2)
        plt.imshow(im[z].clip(-150, 250), cmap='bone')
        plt.contour(preds[i, z], colors=colors_list[i])
        plt.contour(preds[j, z], colors=colors_list[j])
    
        plt.savefig(join(plotp, scan + '_' + str(z) + '.png'))
        plt.close()
# %%
for i in range(6):
    print(confusion[i,:].astype(int))