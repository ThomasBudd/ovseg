from ovseg.preprocessing.ClassCascadePreprocessing import ClassCascadePreprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i", type=int)
args = parser.parse_args()

prev_stages = 3 * [{'data_name': 'OV04',
                    'preprocessed_name': 'multiclass_1_9',
                    'model_name': 'U-Net5_new_sampling'}] + [{'data_name': 'OV04',
                      'preprocessed_name': 'pod_067',
                      'model_name': 'larger_res_encoder'}]
                    

pref_list = 3 * ['pod_om'] + ['pod']

lb_classes_list = [[2, 13, 15, 17], [2], [1, 2, 9], [1]]


prev_stage = prev_stages[args.i]
pref = pref_list[args.i]
lb_classes = lb_classes_list[args.i]


prep = ClassCascadePreprocessing(apply_resizing=True,
                                 apply_pooling=False,
                                 apply_windowing=True,
                                 save_only_fg_scans=False,
                                 prev_stages=prev_stage,
                                 lb_classes=lb_classes)

prep.plan_preprocessing_raw_data('OV04')

prep.preprocess_raw_data('OV04', 'cascade_'+pref+'_'+'_'.join([str(c) for c in lb_classes]))
