from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.utils.io import read_data_tpl_from_nii
import numpy as np
import matplotlib.pyplot as plt

data_tpl = read_data_tpl_from_nii('BARTS', 283)

model = SegmentationModel(val_fold=6, data_name='OV04', preprocessed_name='pod_quater',
                          model_name='test_prg_trn', is_inference_only=True)

pred = model(data_tpl)
lb = (data_tpl['label'] == 9).astype(float)
print(200 * np.sum(pred * lb) / np.sum(pred + lb))

z = np.argmax(np.sum(lb, (1, 2)))
plt.imshow(data_tpl['image'][z].clip(-150, 250), cmap='gray')
plt.contour(lb[z] > 0, linewidths=0.5, colors='red', linestyles='solid')
plt.contour(pred[z] > 0, linewidths=0.5, colors='blue', linestyles='dashed')

model.save_prediction(data_tpl, 'test_save', 'delete_me.nii.gz')