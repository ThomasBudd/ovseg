from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg import OV_PREPROCESSED
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("raw_data", default='OV04', nargs='+',)
args = parser.parse_args()

raw_data = args.raw_data
data_name = '_'.join(sorted(raw_data))

lb_classes_list = [[1, 9], [1, 2, 3, 5, 6, 7], [13, 14, 15, 17]]
p_name_list = ['pod_om', 'abdominal_lesions', 'lymph_nodes']
target_spacing_list = [[5.0, 0.8, 0.8], [5.0, 0.8, 0.8], [5.0, 0.67, 0.67]]

for lb_classes, p_name, target_spacing in zip(lb_classes_list,
                                              p_name_list,
                                              target_spacing_list):
    prep = SegmentationPreprocessing(apply_resizing=True,
                                     apply_pooling=False,
                                     apply_windowing=True,
                                     lb_classes=lb_classes,
                                     target_spacing=target_spacing,
                                     save_only_fg_scans=False)

    prep.plan_preprocessing_raw_data(raw_data)
    prep.preprocess_raw_data(raw_data, p_name)

print('Converted the following datasets:')
print(raw_data)
print('The preprocessed data is stored under the name '+data_name)
print(data_name)