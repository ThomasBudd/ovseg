from ovseg.preprocessing.RegionfindingPreprocessing import RegionfindingPreprocessing


prep = RegionfindingPreprocessing(apply_resizing=True,
                                  apply_pooling=False,
                                  apply_windowing=True,
                                  z_to_xy_ratio=8,
                                  r=13,
                                  reduce_lb_to_single_class=True)

prep.plan_preprocessing_raw_data('OV04')

prep.preprocess_raw_data('OV04', 'bin_reg')

