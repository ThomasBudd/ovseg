from ovseg.preprocessing.RegionfindingPreprocessing import RegionfindingPreprocessing

prep = RegionfindingPreprocessing(apply_resizing=True,
                                  apply_pooling=False,
                                  apply_windowing=True,
                                  z_to_xy_ratio=8,
                                  r=13,
                                  lb_classes=[1, 2, 9, 13, 15, 17],
                                  save_only_fg_scans=False)

prep.plan_preprocessing_raw_data('OV04')

prep.preprocess_raw_data('OV04', 'multiclass_reg')

