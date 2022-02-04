import numpy as np
from ovseg.utils.torch_morph import eros_2d
from ovseg.data.Dataset import raw_Dataset
from torch.nn.functional import interpolate
from ovseg.utils.seg_fg_dial import seg_eros, seg_fg_dial
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

r_max = 15

def edge_coordinates(seg):
    
    seg_gpu = torch.from_numpy(seg[np.newaxis]).type(torch.float)
    seg_eros = eros_2d(seg_gpu).cpu().numpy()[0]
    return np.stack(np.where((seg - seg_eros).astype(int)), -1)

def max_dist(coords1, coords2):

    if coords1.shape[0] == 0 or coords2.shape[0] == 0:
        return 0

    s = target_spacing.reshape((1,3))
    
    dists = [np.min(np.sqrt(np.sum(((coords1[i:i+1] - coords2) * s)**2, 1))) for i in range(coords1.shape[0])]

    return np.max(dists)        

def compute_r1_r2(label, pred):
            
    ovlp = label * pred
    
    # compute r1    
    lb_coords = edge_coordinates(label - ovlp)
    pred_coords = edge_coordinates(pred)
    
    r1 = max_dist(lb_coords, pred_coords)
    r1 = np.ceil(r1 / target_spacing[1])
    
    # compute r2
    lb_coords = edge_coordinates(label)
    pred_coords = edge_coordinates(pred - ovlp)
    
    r2 = max_dist(pred_coords, lb_coords)
    r2 = np.ceil(r2 / target_spacing[1])
        
    return r1, r2

def compute_sens_prec(label, pred, r1, r2):
    
    z_to_xy_ratio = target_spacing[0]/target_spacing[1]
    
    pred_dial = seg_fg_dial(pred, r1, z_to_xy_ratio=z_to_xy_ratio, use_3d_ops=True)
    pred_eros = seg_eros(pred, r2, z_to_xy_ratio=z_to_xy_ratio, use_3d_ops=True)
    
    sens = np.sum(label * pred_dial) / np.sum(label)
    prec = np.sum(label * pred_eros) / np.sum(pred_eros)
    
    return sens, prec


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
sp_list = []

for i in tqdm(range(len(ds))):
    data_tpl = ds[i]
    label, pred = get_label_pred(data_tpl)
    
    r1, r2 = compute_r1_r2(label, pred)
    r_list.append(compute_r1_r2(label, pred))
    
    sp_list.append(compute_sens_prec(label, pred, r1, r2))
    
    if i == 10:
        break


print(r_list)
np.save(os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'kits21',
                     'r_list_v2'),
        r_list)
np.save(os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'kits21',
                     'sp_list'),
        sp_list)