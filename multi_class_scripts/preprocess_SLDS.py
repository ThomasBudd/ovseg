from ovseg.preprocessing.SLDSPreprocessing import SLDSPreprocessing


prep = SLDSPreprocessing(apply_resizing=True,
                         apply_pooling=False,
                         apply_windowing=True,
                         r=13,
                         z_to_xy_ratio=5.0/0.67)

prep.plan_preprocessing_raw_data('OV04')
prep.preprocess_raw_data('OV04', 'SLDS')
