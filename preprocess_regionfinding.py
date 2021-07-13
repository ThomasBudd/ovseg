from ovseg.preprocessing.RegionfindingPreprocessing import RegionfindingPreprocessing

prep = RegionfindingPreprocessing(True, False, True, mask_dist=[2, 15, 15])
prep.plan_preprocessing_raw_data('OV04')
prep.preprocess_raw_data('OV04', 'multiclass')