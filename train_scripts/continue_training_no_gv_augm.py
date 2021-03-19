from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
import os

model_params = get_model_params_2d_segmentation()
model_params['training']['num_epochs'] = 500
model_params['training']['opt_params']['lr'] = 0.005
del model_params['augmentation']['GPU_params']['grayvalue']

model = SegmentationModel(val_fold=0, data_name='OV04',
                          model_name='pretrained_continued_no_gv_aug',
                          model_parameters=model_params)

path = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', 'OV04',
                    'pretrained_segmentation', 'pretrained_segmentation')
model.training.load_last_checkpoint(path)
model.training.train()
model.eval_validation_set()
model.eval_training_set()