from ovseg.model.ClassSegmentationModel import ClassSegmentationModel
from time import sleep

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()


model = ClassSegmentationModel(val_fold=args.vf, data_name='OV04',
                               preprocessed_name='ClassSegmentation',
                               model_name='U-Net5')

model.eval_validation_set()
model.eval_raw_data_npz('BARTS')
