import os
from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel

if not os.path.exists(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04_test')):
    preprocessing = SegmentationPreprocessing(use_only_classes=[9])
    preprocessing.plan_preprocessing_raw_data('OV04_test')
    preprocessing.preprocess_raw_data('OV04_test')

model_params = get_model_params_2d_segmentation()


data_name = 'OV04_test'

model_params['network']['filters'] = 8
model_params['training']['num_epochs'] = 50

for val_fold in [0, 1, 3]:
    model_pretrain = SegmentationModel(val_fold=val_fold,
                                       data_name=data_name,
                                       model_name='test_model',
                                       model_parameters=model_params)
    model_pretrain.training.train()
    model_pretrain.eval_validation_set()
    model_pretrain.eval_training_set()
