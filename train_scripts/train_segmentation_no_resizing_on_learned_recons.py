from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--angles', required=False, default=2500)
parser.add_argument('-n', '--n_reps', required=False, default=3)
model_params = get_model_params_2d_segmentation()
args = parser.parse_args()

val_fold = 0

data_name = 'OV04'
preprocessed_name = 'pod_no_resizing'
model_params['data']['trn_dl_params']['store_coords_in_ram'] = True
model_params['data']['val_dl_params']['store_coords_in_ram'] = True
model_params['data']['folders'][0] = 'recon_fbp_convs_{}_eights_8_32'.format(args.angles)

for i in range(args.n_reps):
    model_name = 'segmentation_no_resizing_{}_{}'+str(args.angles, i)

    model = SegmentationModel(val_fold=val_fold,
                              data_name=data_name,
                              model_name=model_name,
                              model_parameters=model_params,
                              preprocessed_name=preprocessed_name)
    model.training.train()
    model.eval_validation_set(save_preds=False)
    model.eval_training_set(save_preds=False)
