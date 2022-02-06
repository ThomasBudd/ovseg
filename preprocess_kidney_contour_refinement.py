from ovseg.preprocessing.ContourRefinementV2Preprocessing import ContourRefinementV2Preprocessing


prev_stages = {'data_name': 'kits21',
               'preprocessed_name': 'kidney_low',
               'model_name': 'first_try'}

preprocessing = ContourRefinementV2Preprocessing(apply_resizing=True,
                                               apply_pooling=False,
                                               apply_windowing=True,
                                               reduce_lb_to_single_class=True,
                                               prev_stages=prev_stages,
                                               target_spacing = [3.0, 0.78125, 0.78125],
                                               window = [-61.5, 309.5],
                                               scaling = [ 73.998665, 104.83186 ],
                                               r_dial=10)

preprocessing.preprocess_raw_data(raw_data='kits21',
                                  preprocessed_name='kidney_full_refine')
