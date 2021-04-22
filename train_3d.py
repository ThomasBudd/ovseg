from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("fold")

args = parser.parse_args()

model_params = get_model_params_3d_nnUNet([56, 192, 160], 2)

model = SegmentationModel(val_fold=int(args.fold),
                          data_name='OV04',
                          preprocessed_name='pod_half',
                          model_name='pod_half_benchmark_2',
                          model_parameters=model_params)

model.training.train()
model.eval_validation_set(save_preds=False)
model.eval_training_set()
