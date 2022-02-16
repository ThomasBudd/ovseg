from ovseg.preprocessing.RegionexpertPreprocessing import RegionexpertPreprocessing

prep = RegionexpertPreprocessing(apply_resizing=True,
                                 apply_pooling=False,
                                 apply_windowing=True,
                                 region_finding_model={'data_name': 'OV04_test',
                                                       'preprocessed_name': 'pod_om_reg',
                                                       'model_name': 'regfinding_0.1'},
                                 lb_classes=[9])

prep.plan_preprocessing_raw_data('OV04_test')
prep.preprocess_raw_data('OV04_test', 'pod_reg_expert')
