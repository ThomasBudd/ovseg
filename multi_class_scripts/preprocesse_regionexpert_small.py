from ovseg.preprocessing.RegionexpertPreprocessing import RegionexpertPreprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("reg", type=int)
args = parser.parse_args()

if args.reg == 0:

    prep = RegionexpertPreprocessing(apply_resizing=True,
                                     apply_pooling=False,
                                     apply_windowing=True,
                                     region_finding_model={'data_name': 'OV04',
                                                           'preprocessed_name': 'mesentery_reg',
                                                           'model_name': 'regfinding_0.1'},
                                     lb_classes=[5],
                                     target_spacing=[5.0, 0.67, 0.67],
                                     save_only_fg_scans=False)
    
    prep.plan_preprocessing_raw_data('OV04')
    prep.preprocess_raw_data('OV04', 'mesentery_reg_expert')

else:
    prep = RegionexpertPreprocessing(apply_resizing=True,
                                     apply_pooling=False,
                                     apply_windowing=True,
                                     region_finding_model={'data_name': 'OV04',
                                                           'preprocessed_name': 'small_reg',
                                                           'model_name': 'regfinding_0.1'},
                                     lb_classes=[3,4,5,6,7,11,12,13,14,15,16,17,18],
                                     reduce_lb_to_single_class=True,
                                     target_spacing=[5.0, 0.67, 0.67],
                                     save_only_fg_scans=False)
    
    prep.plan_preprocessing_raw_data('OV04')
    prep.preprocess_raw_data('OV04', 'small_reg_expert')
