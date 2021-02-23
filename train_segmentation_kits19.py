from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pretrain_only", required=False, default=False, action='store_true')
model_params = get_model_params_2d_segmentation()
args = parser.parse_args()

val_fold = 0

data_name = 'kits19'
model_params['data']['n_folds'] = 3
model_params['network']['out_channels'] = 3
if args.pretrain_only:
    del model_params['augmentation']['GPU_params']['grayvalue']
    model_params['training']['num_epochs'] = 500
    model_params['training']['lr_params']['lr_min'] = 0.01 * 0.5**0.9
    model_name = 'segmentation_pretrain'
else:
    model_name = 'segmentation_fully_trained'

model_pretrain = SegmentationModel(val_fold=0, data_name=data_name,
                                   model_name=model_name,
                                   model_parameters=model_params)
model_pretrain.training.train()
model_pretrain.validate()
