from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel
import os
from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing

data_name = 'OV04'
preprocessed_name = 'pod_no_resizing'

if not os.path.exists(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', data_name,
                                   preprocessed_name)):
    preprocessing = SegmentationPreprocessing(use_only_classes=[9], apply_resizing=False)
    preprocessing.plan_preprocessing_raw_data('OV04')
    preprocessing.preprocess_raw_data('OV04', preprocessed_name=preprocessed_name)


model_params = get_model_params_2d_segmentation()
model_params_pretrained = get_model_params_2d_segmentation()

val_fold = 0

model_params_pretrained['training']['num_epochs'] = 500
model_params_pretrained['training']['lr_params']['lr_min'] = 0.01 * 0.5**0.9
model_name_pretrained = 'segmentation_pod_no_resizing_pretrained'
model_name = 'segmentation_pod_no_resizing'

model_pretrain = SegmentationModel(val_fold=val_fold,
                                   data_name=data_name,
                                   model_name=model_name_pretrained,
                                   model_parameters=model_params_pretrained,
                                   preprocessed_name=preprocessed_name)
model_pretrain.training.train()
model_pretrain.eval_validation_set(save_preds=True)
model_pretrain.eval_training_set(save_preds=False)

model = SegmentationModel(val_fold=val_fold,
                          data_name=data_name,
                          model_name=model_name,
                          model_parameters=model_params,
                          preprocessed_name=preprocessed_name)
model.training.load_last_checkpoint(model_pretrain.model_path)
model.training.train()
model.eval_validation_set(save_preds=True)
model.eval_training_set(save_preds=False)
