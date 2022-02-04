import numpy as np
from ovseg.data.Dataset import raw_Dataset
from torch.nn.functional import interpolate
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'kits21', 'kidney_low')
if not os.path.exists(plotp):
    os.makedirs(plotp)

prev_stage = {'data_name': 'kits21',
              'preprocessed_name': 'kidney_low',
              'model_name': 'first_try'}
pred_key = '_'.join(['prediction',
                     prev_stage['data_name'],
                     prev_stage['preprocessed_name'],
                     prev_stage['model_name']])


ds = raw_Dataset('kits21', prev_stages=prev_stage)

for i in tqdm(range(len(ds))):
    
    data_tpl = ds[i]
    
    im = data_tpl['image'].clip(-150, 250)
    lb = (data_tpl['label'] > 0).astype(float)
    pred = data_tpl[pred_key]
    
    contains = np.where(np.sum(lb + pred, (1,2)))[0]
    
    for z in contains:
        plt.imshow(im[z],cmap='bone')
        if lb[z].max() > 0:
            plt.contour(lb[z] > 0,colors='red')
        if pred[z].max() > 0:
            plt.contour(pred[z] > 0,colors='blue')
        plt.axis('off')
        plt.savefig(os.path.join(plotp, data_tpl['scan'] + '_' + str(z)))
        plt.close()