from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run", type=int)
args = parser.parse_args()


im_folder = ['restaurations_full', 'restaurations_half', 'restaurations_quater',
             'restaurations_eights', 'restaurations_16', 'restaurations_32'][args.exp]

model_params = get_model_params_2d_segmentation()
model_params['data']['folders'][0] = im_folder
model_params['network']['norm'] = 'inst'
model_params['network']['norm_params'] = {'affine': True, 'eps': 1e-2}
model_params['training']['opt_name'] = 'ADAM'
model_params['training']['opt_params'] = {'lr': 10**-4}
model_name = '2d_sequential_new_'+im_folder[14:]

for i in range(3):
    model = SegmentationModel(val_fold=5+str(i),
                              data_name='OV04',
                              model_name=model_name,
                              model_parameters=model_params,
                              preprocessed_name='pod_2d')
    
    model.training.train()
    model.eval_validation_set()
