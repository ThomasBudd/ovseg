from ovseg.utils.io import load_pkl, save_pkl, save_txt
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('i', type=int)
args = parser.parse_args()


data_name = 'OV04'
preprocessed_name = 'pod_067'
model_name = 'larger_res_encoder_simple_prediction'

path_to_params = os.path.join(os.environ['OV_DATA_BASE'],
                              'trained_models',
                              data_name,
                              preprocessed_name,
                              model_name,
                              'model_parameters.pkl')
model_params = load_pkl(path_to_params)
model_params['prediction']['mode'] = 'simple'
save_pkl(model_params, path_to_params)
save_txt(model_params, path_to_params[:-4])

val_fold = list(range(5)) if args.i == 0 else [5,6,7]

ens = SegmentationEnsemble(val_fold=val_fold,
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=preprocessed_name)

ens.eval_raw_dataset('BARTS', force_evaluation=True)
