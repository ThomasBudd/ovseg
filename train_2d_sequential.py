from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
import nibabel as nib
import numpy as np
from os import environ, mkdir, listdir
from os.path import join, exists
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
parser.add_argument("run", type=int)
args = parser.parse_args()

im_folder = ['images', 'restaurations_full', 'restaurations_half', 'restaurations_quater',
             'restaurations_eights', 'restaurations_16', 'restaurations_32'][args.exp]


model_params = get_model_params_2d_segmentation()
model_params['data']['folders'][0] = im_folder
model_params['network']['norm'] = 'inst'
model_params['network']['filters'] = 8

if args.run == -1:
    for i in range(3):
        model = SegmentationModel(val_fold=5+i,
                                  data_name='OV04',
                                  model_name='2d_sequential_'+im_folder+'_small',
                                  model_parameters=model_params,
                                  preprocessed_name='pod_2d')
        
        model.training.train()
        model.eval_validation_set()
else:
    model = SegmentationModel(val_fold=5+args.run,
                              data_name='OV04',
                              model_name='2d_sequential_'+im_folder+'_small',
                              model_parameters=model_params,
                              preprocessed_name='pod_2d')
    
    model.training.train()
    model.eval_validation_set()
