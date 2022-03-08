from ovseg.preprocessing.SensPrecPreprocessing import SensPrecPreprocessing


prep = SensPrecPreprocessing(apply_resizing=True,
                             apply_pooling=False,
                             apply_windowing=True,
                             z_to_xy_ratio=8,
                             r=6,
                             lb_classes=[1])

prep.plan_preprocessing_raw_data('OV04_test')

prep.preprocess_raw_data('OV04_test', 'sens_prec_reg')

# %%

# import numpy as np
# import matplotlib.pyplot as plt
# im = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04_test\\sens_prec_reg\\images\\case_000.npy').astype(float)
# lb = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04_test\\sens_prec_reg\\labels\\case_000.npy').astype(float)
# reg = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04_test\\sens_prec_reg\\regions\\case_000.npy').astype(float)

# z = 66
# plt.imshow(im[0, z], cmap='bone')
# plt.contour(lb[0, z], colors='b')
# # plt.contour(reg[0, z] == 2, colors='r')
# # plt.contour(reg[0, z] == 1, colors='r')

# from ovseg.utils.seg_fg_dial import seg_fg_dial, seg_eros
# lb_eros = seg_eros(lb[0], r=6, z_to_xy_ratio=8)

# plt.contour(lb_eros[z], colors='g')
