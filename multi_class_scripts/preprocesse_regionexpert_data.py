from ovseg.preprocessing.RegionexpertPreprocessing import RegionexpertPreprocessing
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

classes = [2, 15]

prefs = ['liver', 'diaph']

w = 0.3

for cl, pref in zip(classes, prefs):
    prep = RegionexpertPreprocessing(apply_resizing=True,
                                     apply_pooling=False,
                                     apply_windowing=True,
                                     region_finding_model={'data_name': 'OV04',
                                                           'preprocessed_name': 'multiclass_reg',
                                                           'model_name': 'regfinding_{}'.format(w)},
                                     lb_classes=[cl],
                                     target_spacing=[5.0, 0.8, 0.8])
    
    prep.plan_preprocessing_raw_data('OV04')
    prep.preprocess_raw_data('OV04', pref+'_reg_expert')
