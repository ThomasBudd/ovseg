from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("run", type=int)
parser.add_argument("exp", type=int)
args = parser.parse_args()


im_folder = ['restaurations_full', 'restaurations_quater'][args.exp]

model_params = get_model_params_2d_segmentation()
model_params['data']['folders'][0] = im_folder
model_params['network']['norm'] = 'inst'
model_params['network']['norm_params'] = {'affine': True, 'eps': 1e-2}
model_params['training']['opt_name'] = 'ADAM'
model_params['training']['opt_params'] = {'lr': 10**-4}
model_params['training']['num_epochs'] = 500
model_name = '2d_sequential_refine_'+im_folder[14:]


model = SegmentationModel(val_fold=5+args.run,
                          data_name='OV04',
                          model_name=model_name,
                          model_parameters=model_params,
                          preprocessed_name='pod_2d')

path_to_weights = os.path.join(os.environ['OV_DATA_BASE'],
                               'trained_models',
                               'OV04',
                               'pod_2d',
                               '2d_num_epochs_ADAM_long',
                               'fold_{}'.format(5+args.run),
                               'network_weights_2000')
model.network.load_state_dict(torch.load(path_to_weights,
                                 map_location=torch.device('cuda')))
model.training.train()
model.eval_validation_set()
