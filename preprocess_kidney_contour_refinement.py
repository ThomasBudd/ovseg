from ovseg.preprocessing.ContourRefinementPreprocessing import ContourRefinementPreprocessing


prev_stages = {'data_name': 'kits21',
               'preprocessed_name': 'kidney_low',
               'model_name': 'first_try'}

preprocessing = ContourRefinementPreprocessing(apply_resizing=True,
                                               apply_pooling=False,
                                               apply_windowing=True,
                                               reduce_lb_to_single_class=True,
                                               prev_stages=prev_stages)

preprocessing.plan_preprocessing_raw_data('kits21')

preprocessing.preprocess_raw_data(raw_data='kits21',
                                  preprocessed_name='kidney_full')
