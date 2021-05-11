from ovseg.model.model_parameters_segmentation import get_model_params_3d_nfUNet
from ovseg.model.SegmentationModel import SegmentationModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i")

args = parser.parse_args()

model_params = get_model_params_3d_nfUNet([48, 192, 192], 2, True)

model = SegmentationModel(val_fold=int(args.i), data_name='OV04', model_name='nfUNet',
                          preprocessed_name='pod_half', model_parameters=model_params)

model.training.train()
model.eval_validation_set(False)
