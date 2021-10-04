from ovseg.preprocessing.SLDSExpertPreprocessing import SLDSExpertPreprocessing
import argparse


prep = SLDSExpertPreprocessing(apply_resizing=True,
                               apply_pooling=False,
                               apply_windowing=True,
                               region_finding_model={'data_name': 'OV04',
                                                     'preprocessed_name': 'SLDS',
                                                     'model_name': 'U-Net5_0.01'},
                               lb_classes=[2])

prep.plan_preprocessing_raw_data('OV04')
prep.preprocess_raw_data('OV04', 'SLDS_reg_expert_0.01')
