from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
import os
import numpy as np

preprocessing = SegmentationPreprocessing(apply_pooling=True,
                                          apply_resizing=True,
                                          apply_windowing=True,
                                          pooling_stride=[1, 4, 4],
                                          lb_classes=[9])

preprocessing.plan_preprocessing_raw_data('OV04')

preprocessing.preprocess_raw_data('OV04', 'pod_quater')
