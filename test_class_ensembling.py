from ovseg.data.Dataset import raw_Dataset
from ovseg.preprocessing.ClassEnsemblePreprocessing import ClassEnsemblePreprocessing
import os
import matplotlib.pyplot as plt
import numpy as np

# %% test the raw dataset
prev_stages = [{'data_name': 'OV04',
                'preprocessed_name': 'om_08',
                'model_name':'res_encoder_no_prg_lrn'},
               {'data_name': 'OV04',
                'preprocessed_name': 'pod_067',
                'model_name':'larger_res_encoder'}]
pred_folders = [os.path.join(os.environ['OV_DATA_BASE'], 'predictions', ps['data_name'],
                             ps['preprocessed_name'], ps['model_name'], 'cross_validation')
                for ps in prev_stages]

scans = [scan.split('.')[0] for scan in os.listdir(pred_folders[0]) if scan in os.listdir(pred_folders[1])]
raw_path = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04')

ds = raw_Dataset(raw_path, scans=scans, prev_stages=prev_stages)
data_tpl = ds[0]

# %%
preprocessing = ClassEnsemblePreprocessing(prev_stages,
                                           apply_resizing=True,
                                           apply_pooling=False,
                                           apply_windowing=True,
                                           window=[-150, 250],
                                           scaling=[1, 0],
                                           target_spacing=[5.0, 1.0, 1.0])

xb = preprocessing(data_tpl).cpu().numpy()

z = np.argmax(np.sum(xb[2] == 1, (1, 2)))
im, pred, lb = xb[:, z]
plt.imshow(im, cmap='bone')
plt.contour(pred > 0, colors='blue')
plt.contour(lb > 0, colors='red')
