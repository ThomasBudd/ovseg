from ovseg.model.RegionexpertModel import RegionexpertModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import numpy as np
import matplotlib.pyplot as plt

params = get_model_params_3d_res_encoder_U_Net([32, 256, 256], 8)

params['data']['folders'] = ['images', 'labels', 'regions']
params['data']['keys'] = ['image', 'label', 'region']

model = RegionexpertModel(val_fold=0, data_name='OV04_test', preprocessed_name='pod_reg_expert', 
                         model_name='delete_meee', model_parameters=params)

# %%

for j in range(5):
    for batch in model.data.trn_dl:
        break
    
    batch = batch.cpu().numpy().astype(np.float32)
    
    for i in range(2):
        plt.subplot(2, 5, j+1 + 5*i)
        z = np.argmax(np.sum(batch[i, 1], (1, 2)))
        im = batch[i, 0, z, 128:-128, 128:-128]
        reg = batch[i, 1, z, 128:-128, 128:-128]
        if reg.max() == 0:
            raise ValueError('HALT STOPP')
        lb = batch[i, 2, z, 128:-128, 128:-128]
        plt.imshow(im, cmap='bone')
        plt.contour(reg > 0, colors='red', linewidths=0.5)
        plt.contour(lb > 0, colors='red', linewidths=1)
