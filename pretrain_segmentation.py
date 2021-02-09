from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel
import torch

val_fold = 0
data_name = 'OV04'

# %% first segmentation on siemens reconstructions
model_params = get_model_params_2d_segmentation()

model_params['training']['num_epochs'] = 500
model_params['training']['lr_params']['lr_min'] = 0.5**0.9 * 0.01

model = SegmentationModel(val_fold=val_fold, data_name=data_name, model_parameters=model_params,
                          model_name='segmentation_pretrained')
model.training.train()
model.eval_validation_set()
model.eval_training_set()
torch.cuda.empty_cache()
