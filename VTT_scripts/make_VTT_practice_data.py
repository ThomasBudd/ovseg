import os
import pydicom
import numpy as np
from ovseg.utils.io import save_dcmrt_from_data_tpl, read_dcms
from shutil import copytree
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
from ovseg.utils.torch_morph import morph_cleaning
import torch
from skimage.measure import label

np.random.seed(15031993)


tarp = 'D:\\PhD\\Data\\VTT_practise'
tasks = ['omentum', 'pelvic_ovarian']
lb_classes = [1, 9]


def remove_small_ccs(seg):
    
    mask = np.ones_like(seg)
    
    for z in range(seg.shape[0]):
        
        comps = label(seg[z])
        n_comps = comps.max()
        
        for i in range(1, n_comps+1):
            comp = comps == i
    
            if np.sum(comp.astype(float)) <= 10:
                mask[z][comp] = 0
        

    return seg * mask


def clean_label(seg):
    return remove_small_ccs(morph_cleaning(torch.from_numpy(seg).cuda()).cpu().numpy())


for task, cl in zip(tasks, lb_classes):
    
    taskp = os.path.join(tarp, task)
    print(task)
    sleep(0.1)
    
    for i, scan in enumerate(tqdm(os.listdir(taskp))):
        
        data_tpl = read_dcms(os.path.join(taskp, scan))
        seg = (data_tpl['label'] == cl).astype(float)
        
        if i > 1:
            seg = clean_label(seg)
    
        data_tpl['label'] = seg
        
        save_dcmrt_from_data_tpl(data_tpl,
                                 out_file=data_tpl['raw_label_file'],
                                 key='label',
                                 names=[str(cl)],
                                 colors=[[255, 0, 0]])