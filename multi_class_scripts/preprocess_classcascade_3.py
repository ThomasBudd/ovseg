from ovseg.preprocessing.ClassCascadePreprocessing import ClassCascadePreprocessing


prev_stage = [{'data_name': 'OV04',
               'preprocessed_name': 'cascade_lymph_to_rest',
               'model_name': 'U-Net5_old_sampling'}]

lb_classes = [1]
prev_pred_classes = [2, 13, 15, 17]
prep = ClassCascadePreprocessing(apply_resizing=True,
                                 apply_pooling=False,
                                 apply_windowing=True,
                                 save_only_fg_scans=False,
                                 prev_stages=prev_stage,
                                 prev_pred_classes=prev_pred_classes,
                                 lb_classes=lb_classes)

prep.plan_preprocessing_raw_data('OV04')

prep.preprocess_raw_data('OV04', 'cascade_small_to_om')
