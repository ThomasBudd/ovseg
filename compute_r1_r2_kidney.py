import numpy as np
from ovseg.utils.seg_fg_dial import seg_eros, seg_fg_dial
from ovseg.data.Dataset import raw_Dataset
from torch.nn.functional import interpolate
import torch
from tqdm import tqdm
import os

prev_stage = {'data_name': 'kits21',
              'preprocessed_name': 'kidney_low',
              'model_name': 'first_try'}
pred_key = '_'.join(['prediction',
                     prev_stage['data_name'],
                     prev_stage['preprocessed_name'],
                     prev_stage['model_name']])


ds = raw_Dataset('kits21', prev_stages=prev_stage)

target_spacing = np.array([3.0, 0.8, 0.8])
z_to_xy_ratio = 3.0/0.8

r_max = 30


def compute_r1_r2(label, pred):
            
    r1 = 0
    
    vol_lb = np.sum(label)
    ovlp = np.sum(label * pred)
    sens = ovlp / vol_lb
    
    if sens < 1:
        
        while sens < 1 and r1 < r_max:
            r1 += 1
            pred_dial = seg_fg_dial(pred, r1, z_to_xy_ratio=z_to_xy_ratio, use_3d_ops=True)
            ovlp = np.sum(label * pred_dial)
            sens = ovlp / vol_lb
    
    r2 = 0
    
    prec = np.sum(label * pred) / np.sum(pred)
    if prec < 1:
        while prec < 1 and r2 < r_max:
            r2 += 1
            pred_eros = seg_eros(pred, r2, z_to_xy_ratio=z_to_xy_ratio, use_3d_ops=True)
            vol_pred = np.sum(pred_eros)
            if vol_pred == 0:
                prec = 1
            else:
                prec = np.sum(pred_eros * label) / vol_pred
    
        
    return r1, r2

def get_label_pred(data_tpl):
    # reduce to binary label
    lb = (data_tpl['label'] > 0).astype(float)
    pred = data_tpl[pred_key]
    
    # compute rescaling factor
    spacing = data_tpl['spacing']
    scale_factor = (spacing / target_spacing).tolist()
    
    # now do the interpolation
    batch = np.stack([lb, pred])[np.newaxis]
    batch = torch.from_numpy(batch).cuda()
    # nearest neighbour interpolation
    batch = interpolate(batch, scale_factor=scale_factor).cpu().numpy()
    
    return batch[0,0], batch[0, 1]

r_list = []

for i in tqdm(range(len(ds))):
    data_tpl = ds[i]
    label, pred = get_label_pred(data_tpl)
    
    r_list.append(compute_r1_r2(label, pred))
    
    #break


print(r_list)
np.save(os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'kits21',
                     'r_list'),
        r_list)