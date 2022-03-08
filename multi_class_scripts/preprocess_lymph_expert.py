from ovseg.preprocessing.RegionexpertPreprocessing import RegionexpertPreprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("w", type=int)
args = parser.parse_args()


w_list = [2/67, 2/129]#[2/3, 2/5, 2/9, 2/17, 2/33] 
w = w_list[args.w]

prep = RegionexpertPreprocessing(apply_resizing=True,
                                 apply_pooling=False,
                                 apply_windowing=True,
                                 region_finding_model={'data_name': 'OV04',
                                                       'preprocessed_name': 'lymph_reg',
                                                       'model_name': 'regfinding_'+str(w)},
                                 lb_classes=[13, 15, 17],
                                 target_spacing=[5.0, 0.67, 0.67],
                                 save_only_fg_scans=False)

prep.plan_preprocessing_raw_data('OV04')
prep.preprocess_raw_data('OV04', 'lymph_reg_expert_'+str(w))