from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("fold")

args = parser.parse_args()

model_params = get_model_params_3d_nnUNet([48, 192, 192], 1)

model = SegmentationModel(val_fold=int(args.fold),
                          data_name='OV04',
                          preprocessed_name='pod_half',
                          model_name='pod_half_default',
                          model_parameters=model_params)

model.training.train()
model.eval_validation_set(save_preds=False)
model.eval_training_set()
