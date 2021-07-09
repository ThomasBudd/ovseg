from ovseg.preprocessed.SegmentationPreprocessing import SegmentationPreprocessing
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

if args.exp < 4:
    target_spacing = [5.0, 0.8, 0.8]
else:
    target_spacing = [5.0, 0.67, 0.67]

lb_classes = [[1], [2, 3, 4], [5, 6, 7], [13, 14, 15], [9]]

all_classes = [1, 2, 3, 4,5, 6, 7, 9, 13, 14, 15]
mask_classes = [c for c in all_classes if c not in lb_classes]

p_name = ['om_mask', 'lesions_upper_mask', 'lesions_center_mask', 'lesions_lymphnodes_mask',
          'pod_mask']

preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                          apply_pooling=False,
                                          apply_windowing=True,
                                          target_spacing=target_spacing,
                                          lb_classes=lb_classes,
                                          mask_classes=mask_classes,
                                          reduce_lb_to_single_class=True,
                                          save_only_fg_scans=True)


preprocessing.plan_preprocessing_raw_data('OV04',
                                          force_planning=True)

preprocessing.preprocess_raw_data(raw_data='OV04',
                                  preprocessed_name=args.preprocessed_name,
                                  data_name=args.data_name,
                                  save_as_fp16=not args.save_as_fp32)
