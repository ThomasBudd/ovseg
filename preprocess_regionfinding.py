from ovseg.preprocessing.RegionfindingPreprocessing import RegionfindingPreprocessing

prep = RegionfindingPreprocessing(True, False, True)
prep.plan_preprocessing_raw_data('OV04')
prep.preprocess_raw_data('OV04', 'multiclass')