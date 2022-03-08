from ovseg.preprocessing.SLDSExpertPreprocessing import SLDSExpertPreprocessing
import argparse


w_list = [0.001, 0.1]


for w in w_list:
    prep = SLDSExpertPreprocessing(apply_resizing=True,
                                   apply_pooling=False,
                                   apply_windowing=True,
                                   region_finding_model={'data_name': 'OV04',
                                                         'preprocessed_name': 'SLDS',
                                                         'model_name': 'U-Net5_{}'.format(w)},
                                   lb_classes=[2],
                                   target_spacing=[5.0, 0.67, 0.67])
    
    prep.plan_preprocessing_raw_data('OV04')
    prep.preprocess_raw_data('OV04', 'SLDS_reg_expert_{}'.format(w))
