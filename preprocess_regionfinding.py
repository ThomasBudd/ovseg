from ovseg.preprocessing.RegionfindingPreprocessing import RegionfindingPreprocessing

prep = RegionfindingPreprocessing(apply_resizing=True,
                                  apply_pooling=False,
                                  apply_windowing=True,
                                  mask_dist=[2, 15, 15])
prep.plan_preprocessing_raw_data('OV04')
prep.preprocess_raw_data('OV04', 'multiclass')