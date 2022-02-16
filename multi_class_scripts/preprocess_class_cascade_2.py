from ovseg.preprocessing.ClassCascadePreprocessing import ClassCascadePreprocessing


prev_stage = [{'data_name': 'OV04',
               'preprocessed_name': 'multiclass_13_15_17',
               'model_name': 'U-Net5_new_sampling'}]

lb_classes = [1, 2, 9]



prep = ClassCascadePreprocessing(apply_resizing=True,
                                 apply_pooling=False,
                                 apply_windowing=True,
                                 save_only_fg_scans=False,
                                 prev_stages=prev_stage,
                                 lb_classes=lb_classes)

prep.plan_preprocessing_raw_data('OV04')

prep.preprocess_raw_data('OV04', 'cascade_lymph_to_rest')
