from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel

val_fold = 0
data_name = 'OV04'

model_names = ['recon_LPD_full_HU', 'recon_fbp_convs_full_HU', 'reconstruction_network_fbp_convs']

for model_name in model_names:
    model_params = get_model_params_2d_segmentation()
    model_params['data']['folders'][0] = model_name

    model = SegmentationModel(val_fold=val_fold, data_name=data_name, model_parameters=model_params,
                              model_name='segmentation_on_'+model_name)