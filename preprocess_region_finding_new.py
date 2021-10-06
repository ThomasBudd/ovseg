from ovseg.preprocessing.RegionfindingPreprocessing import RegionfindingPreprocessing


prep = RegionfindingPreprocessing(apply_resizing=True,
                                  apply_pooling=False,
                                  apply_windowing=True,
                                  z_to_xy_ratio=8,
                                  r=13,
                                  lb_classes=[5],
                                  save_only_fg_scans=False)

prep.plan_preprocessing_raw_data('OV04')

prep.preprocess_raw_data('OV04', 'mesentery_reg')

prep = RegionfindingPreprocessing(apply_resizing=True,
                                  apply_pooling=False,
                                  apply_windowing=True,
                                  z_to_xy_ratio=8,
                                  r=13,
                                  lb_classes=[3,4,5,6,7,11,12,13,14,15,16,17,18],
                                  reduce_lb_to_single_class=True,
                                  save_only_fg_scans=False)

prep.plan_preprocessing_raw_data('OV04')

prep.preprocess_raw_data('OV04', 'small_reg')

