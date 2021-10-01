from ovseg.model.RegionexpertModel import RegionexpertModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

p_name = 'bin_reg_expert_regfinding_U-Net5_0.001'
model_name = 'U-Net5'


model = RegionexpertModel(val_fold=args.exp + 1,
                          data_name='OV04',
                          preprocessed_name=p_name, 
                          model_name=model_name)
model.training.train()
model.eval_validation_set()
model.eval_raw_data_npz('BARTS')
