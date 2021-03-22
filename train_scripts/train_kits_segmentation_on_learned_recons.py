from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel
import argparse

parser = argparse.ArgumentParser()
model_params = get_model_params_2d_segmentation()
args = parser.parse_args()

val_fold = 0

data_name = 'kits19'
preprocessed_name = 'default'
model_params['data']['trn_dl_params']['store_coords_in_ram'] = True
model_params['data']['val_dl_params']['store_coords_in_ram'] = True
model_params['data']['folders'][0] = 'recon_fbp_convs_pretrained_continued'

model_params['network']['out_channels'] = 3
model_name = 'segmentation_on_learned_recons'

model_pretrain = SegmentationModel(val_fold=val_fold,
                                   data_name=data_name,
                                   model_name=model_name,
                                   model_parameters=model_params,
                                   preprocessed_name=preprocessed_name)
model_pretrain.training.train()
model_pretrain.eval_validation_set(save_preds=True)
model_pretrain.eval_training_set(save_preds=False)
