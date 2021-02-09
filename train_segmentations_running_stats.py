from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation

model_params = get_model_params_2d_segmentation()
model_params['data']['trn_dl_params']['min_biased_samples'] = 6
model_params['data']['val_dl_params']['min_biased_samples'] = 6
model_params['training']['num_epochs'] = 500
model_params['network']['norm_params'] = {'affine': True}

data_name = 'OV04'
val_fold = 0

for trs in [True, False]:
    model_name = 'seg_2d_inst_norm_trs' if trs else 'seg_2d_inst_norm'
    model_params['network']['norm_params']['track_running_stats'] = trs
    model = SegmentationModel(val_fold=val_fold, data_name=data_name, model_name=model_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set(False, False)
