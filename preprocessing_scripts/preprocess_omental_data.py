from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing

sp_list = [1.2, 1.0, 0.8, 0.67]
s_list = ['12', '10', '08', '067']

for sp, s in zip(sp_list, s_list):
    target_spacing = [5.0, sp, sp]
    preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                              apply_pooling=False,
                                              apply_windowing=True,
                                              target_spacing=target_spacing,
                                              lb_classes=[1],
                                              lb_min_vol=1000,
                                              save_only_fg_scans=True)
    
    
    preprocessing.plan_preprocessing_raw_data('OV04',
                                              force_planning=True)
    
    preprocessing.preprocess_raw_data(raw_data=['OV04'],
                                      preprocessed_name='om_'+s,
                                      save_as_fp16=True)
