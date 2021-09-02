from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing


prev_stages = [{'data_name': 'OV04',
                'preprocessed_name': 'pod_067',
                'model_name': 'larger_res_encoder'},
               {'data_name': 'OV04',
                'preprocessed_name': 'om_08',
                'model_name': 'res_encoder_no_prg_lrn'}]

preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                          apply_pooling=False,
                                          apply_windowing=True,
                                          lb_classes=[1, 9],
                                          target_spacing=[5.0, 0.8, 0.8],
                                          prev_stages=prev_stages,
                                          save_only_fg_scans=False)


preprocessing.plan_preprocessing_raw_data('OV04',
                                          force_planning=True)

preprocessing.preprocess_raw_data(raw_data='OV04',
                                  preprocessed_name='pod_om_cascade_08',
                                  save_as_fp16=True)


for p in ['lesions_upper', 'lesions_lymphnodes', 'lesions_center']:
    prev_stages.append({'data_name': 'OV04', 'preprocessed_name': p,
                        'model_name': 'res_encoder_no_prg_lrn'})
preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                          apply_pooling=False,
                                          apply_windowing=True,
                                          lb_classes=[1, 9, 2, 3, 4, 5, 6, 7, 13, 14, 15],
                                          target_spacing=[5.0, 0.8, 0.8],
                                          prev_stages=prev_stages,
                                          save_only_fg_scans=False)


preprocessing.plan_preprocessing_raw_data('OV04',
                                          force_planning=True)

preprocessing.preprocess_raw_data(raw_data='OV04',
                                  preprocessed_name='multiclass_cascade_08',
                                  save_as_fp16=True)
