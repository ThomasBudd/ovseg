from ovseg.preprocessing.RegionfindingPreprocessing import RegionfindingPreprocessing


prep = RegionfindingPreprocessing(apply_resizing=True,
                                  apply_pooling=False,
                                  apply_windowing=True,
                                  target_spacing=[5.0, 0.8, 0.8],
                                  z_to_xy_ratio=6.25,
                                  r=13,
                                  lb_classes=[1, 9, 2, 3, 4, 5, 6, 7, 13, 14, 15])

prep.plan_preprocessing_raw_data('OV04')

prep.preprocess_raw_data('OV04', 'multiclass_reg')

