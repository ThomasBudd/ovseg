from ovseg.preprocessing.ClassSegmentationPreprocessing import ClassSegmentationPreprocessing

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

prep = ClassSegmentationPreprocessing(apply_resizing=True,
                                      apply_pooling=False,
                                      apply_windowing=True,
                                      lb_classes=[1, 2, 9, 13, 15, 17],
                                      prev_stages=prev_stages)


prep.plan_preprocessing_raw_data('OV04')

prep.preprocess_raw_data('OV04', 'ClassSegmentation')


# # %%
# import os
# import numpy as np
# import matplotlib.pyplot as plt


# prep = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04_test', 'ClassSegmentation')
# scan = 'case_001.npy'

# for i, scan in enumerate(['case_006.npy', 'case_008.npy']):
#     im = np.load(os.path.join(prep, 'images', scan)).astype(float)
#     lb = np.load(os.path.join(prep, 'labels', scan)).astype(float)
#     pp = np.load(os.path.join(prep, 'prev_preds', scan)).astype(float)
#     print(np.unique(lb))
    
    
#     z = np.argmax(np.sum(lb > 1, (0, 2, 3)))
#     plt.subplot(1,2,i+1)
#     plt.imshow(im[0, z], cmap='bone')
#     plt.contour(lb[0, z] > 1, colors='red')
#     plt.contour(pp[0, z], colors='blue')