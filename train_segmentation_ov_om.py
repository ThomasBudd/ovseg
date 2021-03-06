from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pretrain_only", required=False, default=False, action='store_true')
model_params = get_model_params_2d_segmentation()
args = parser.parse_args()

val_fold = args.fold

data_name = 'Ov04'
preprocessed_name = 'om_default'
model_params['data']['trn_dl_params']['store_coords_in_ram'] = False
model_params['data']['val_dl_params']['store_coords_in_ram'] = False

model_params['network']['out_channels'] = 3
if args.pretrain_only:
    model_params['training']['num_epochs'] = 500
    model_params['training']['lr_params']['lr_min'] = 0.01 * 0.5**0.9
    model_name = 'segmentation_om_pretrain'
else:
    model_name = 'segmentation_om_fully_trained'

model_pretrain = SegmentationModel(val_fold=val_fold,
                                   data_name=data_name,
                                   model_name=model_name,
                                   model_parameters=model_params,
                                   preprocessed_name=preprocessed_name)
model_pretrain.training.train()
model_pretrain.eval_validation_set(save_preds=True)
model_pretrain.eval_training_set(save_preds=False)
