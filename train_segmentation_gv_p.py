from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--p", required=True)
model_params = get_model_params_2d_segmentation()
args = parser.parse_args()

p = float(args.p)

val_fold = 0

data_name = 'OV04'
for key in ['p_noise', 'p_blur', 'p_bright', 'p_contr']:
    model_params['augmentation']['GPU_params']['grayvalue'][key] = p


model_pretrain = SegmentationModel(val_fold=0, data_name=data_name,
                                   model_name='segmentation_gv_p_{:.4f}'.format(p),
                                   model_parameters=model_params)
model_pretrain.training.train()
model_pretrain.validate()
