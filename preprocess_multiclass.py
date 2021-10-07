from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.preprocessing.RegionfindingPreprocessing import RegionfindingPreprocessing

lb_classes_list = [[2], [1, 9], [1, 2, 9], [1, 2, 9, 13, 15, 17], [13, 15, 17]]


for lb_classes in lb_classes_list:
    prep = SegmentationPreprocessing(apply_resizing=True,
                                     apply_pooling=False,
                                     apply_windowing=True,
                                     lb_classes=lb_classes,
                                     save_only_fg_scans=False)
    
    prep.plan_preprocessing_raw_data('OV04')
    
    prep.preprocess_raw_data('OV04', 'multiclass_'+'_'.join([str(c) for c in lb_classes]))


lb_classes = [13, 15, 17]


prep = RegionfindingPreprocessing(apply_resizing=True,
                                  apply_pooling=False,
                                  apply_windowing=True,
                                  z_to_xy_ratio=8,
                                  r=13,
                                  lb_classes=[13, 15, 17],
                                  save_only_fg_scans=False)


prep.plan_preprocessing_raw_data('OV04')

prep.preprocess_raw_data('OV04', 'lymph_reg')