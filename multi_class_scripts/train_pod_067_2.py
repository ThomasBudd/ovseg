from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.utils.io import load_pkl, save_pkl
import argparse
from time import sleep
import os
parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

p_name = 'pod_067'
model_name = 'larger_res_encoder'
data_name='OV04'

path_to_model_params = os.path.join(os.environ['OV_DATA_BASE'],
                                    'trained_models',
                                    data_name,
                                    p_name,
                                    model_name,
                                    'model_parameters.pkl')
model_params = load_pkl(path_to_model_params)
model_params['data']['trn_dl_params']['num_workers'] = 5
model_params['data']['val_dl_params']['num_workers'] = 0
save_pkl(model_params, path_to_model_params)
model = SegmentationModel(val_fold=args.vf,
                          data_name=data_name,
                          preprocessed_name=p_name,
                          model_name=model_name)
model.training.train()
model.eval_raw_data_npz('BARTS')

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=p_name)

while not ens.all_folds_complete():
    sleep(60)

ens.eval_raw_dataset('BARTS')