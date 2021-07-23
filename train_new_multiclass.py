from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("fold", type=int)
args = parser.parse_args()

# %% first train the new multiclass model
model_parmas = get_model_params_3d_res_encoder_U_Net([32, 216, 216], 6.25,  n_fg_classes=6)
model_name = 'res_encoder'

model = SegmentationModel(val_fold=args.fold, data_name='OV04', preprocessed_name='new_multiclass',
                          model_name=model_name, model_parameters=model_parmas)

model.training.train()
model.eval_validation_set()

# %% now the new region finding model
w = [0.5, 0.2, 0.1, 0.05, 0.02][args.fold]

model_params = get_model_params_3d_res_encoder_U_Net([32, 216, 216], 6.25, n_fg_classes=6,
                                                     larger_res_encoder=True)
model_params['training']['loss_params'] = {'loss_names': ['cross_entropy_weighted_bg',
                                                          'dice_loss_weighted'],
                                          'loss_kwargs': [{'weight_bg': w,
                                                           'n_fg_classes': 11},
                                                          {'eps': 1e-5,
                                                           'weight': w}]}
model_params['training']['batches_have_masks'] = True

model = RegionfindingModel(val_fold=args.fold, data_name='OV04',
                           preprocessed_name='new_multiclass_rf',
                           model_name='Regionfinding_'+str(w), model_parameters=model_params)

model.training.train()
model.eval_validation_set(save_preds=True, save_plots=False)
model.eval_raw_dataset('BARTS')
