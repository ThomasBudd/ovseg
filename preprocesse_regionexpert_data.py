from ovseg.preprocessing.RegionexpertPreprocessing import RegionexpertPreprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--om", default=False, action='store_true')
args = parser.parse_args()

cl = 1 if args.om else 9

pref = 'om' if args.om else 'pod'

weights = [0.01, 0.03, 0.1, 0.3]


for w in weights:
    prep = RegionexpertPreprocessing(apply_resizing=True,
                                     apply_pooling=False,
                                     apply_windowing=True,
                                     region_finding_model={'data_name': 'OV04',
                                                           'preprocessed_name': 'pod_om_reg',
                                                           'model_name': 'regfinding_{}'.format(w)},
                                     lb_classes=[cl],
                                     target_spacing=[5.0, 0.8, 0.8])
    
    prep.plan_preprocessing_raw_data('OV04')
    prep.preprocess_raw_data('OV04', pref+'_reg_expert_{}'.format(w))
