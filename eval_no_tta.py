from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.utils.io import load_pkl, save_pkl, save_txt
import os

data_name = 'OV04'
preprocessed_name = 'pod_om'
model_name = 'clara_model_no_tta'

path_to_model_params = os.path.join(os.environ['OV_DATA_BASE'],
                                    'trained_models',
                                    data_name,
                                    preprocessed_name,
                                    model_name,
                                    'model_parameters.pkl')

model_params = load_pkl(path_to_model_params)
model_params['prediction']['mode'] = 'simple'
save_pkl(model_params, path_to_model_params)
save_txt(model_params, path_to_model_params[:-4]+'.txt')

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=preprocessed_name)

ens.eval_raw_dataset('BARTS')
ens.eval_raw_dataset('ApolloTCGA')
