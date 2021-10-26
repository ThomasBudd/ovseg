import numpy as np
from os.path import join, exists
from os import environ
from ovseg.data.Dataset import raw_Dataset
import torch
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from tqdm import tqdm


lb_classes = [1, 2, 9, 13, 15, 17]
prev_stages = [{'data_name':'OV04',
                'preprocessed_name': 'pod_067',
                'model_name': 'larger_res_encoder'},
               {'data_name':'OV04',
                'preprocessed_name': 'om_067',
                'model_name': 'larger_res_encoder'},
               {'data_name': 'OV04',
                'preprocessed_name': 'multiclass_1_2_9',
                'model_name': 'U-Net5',
                'lb_classes': [2]},
               {'data_name': 'OV04',
                'preprocessed_name': 'multiclass_13_15_17',
                'model_name': 'U-Net5'}]

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
    prev_preds = []
    for prev_stage, key in zip(prev_stages[:-1], keys_for_previous_stages[:-1]):
        assert key in data_tpl, 'prediction '+key+' from previous stage missing'
        pred = data_tpl[key]
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        # ensure the array is 4d
        pred = maybe_add_channel_dim(pred)
        
        if 'lb_classes' in prev_stage:
            pred_new = np.zeros_like(pred)
            for cl in prev_stage['lb_classes']:
                pred_new[pred == cl] = cl
            prev_preds.append(pred_new)
        else:
            prev_preds.append((pred > 0).astype(float))
    
    full_pred = np.zeros_like(pred)
    
    for pred in prev_preds[::-1]:
        #full_pred[pred > 0] = pred[pred > 0]
        full_pred = full_pred * (pred == 0).astype(float) + pred
    
    return full_pred

dscs_list = []
# %%
for i in tqdm(range(len(ds))):
    data_tpl = ds[i]
    lb = data_tpl['label']

    pred = get_pred(data_tpl)
    
    dscs = np.zeros(len(lb_classes))
    for j, c in enumerate(lb_classes):
        
        seg_c = (lb == c).astype(float)
        pred_c = (pred == c).astype(float)
        
        if seg_c.max() > 0:
            dscs[j] = 200 * np.sum(seg_c * pred_c) / np.sum(seg_c + pred_c)
        else:
            dscs[j] = np.nan
        
    dscs_list.append(dscs)

dscs_list = np.array(dscs_list)

# %%
mean_dscs = np.nanmean(dscs_list, 0)
for i, c in enumerate(lb_classes):
    print('{}: {:.2f}'.format(c, mean_dscs[i]))