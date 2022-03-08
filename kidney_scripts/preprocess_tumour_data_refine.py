from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2


prev_stage_for_mask = {'data_name': 'kits21',
                       'preprocessed_name': 'kidney_full_refine_refine',
                       'model_name': 'refine_model'}

prev_stage_for_input = {'data_name': 'kits21',
                        'preprocessed_name': 'kidney_tumour',
                        'model_name': 'first_try'}

prep = SegmentationPreprocessingV2(apply_resizing=True,
                                   apply_pooling=False,
                                   apply_windowing=True,
                                   lb_classes=[2],
                                   save_only_fg_scans=False,
                                   scaling=[60.1791, 63.9565],
                                   target_spacing=[3.0, 0.78, 0.78],
                                   window=[-52, 286],
                                   prev_stage_for_mask=prev_stage_for_mask,
                                   prev_stage_for_input=prev_stage_for_input)

# prep.plan_preprocessing_raw_data('kits21')
prep.preprocess_raw_data('kits21', 'kidney_tumour_refine')
