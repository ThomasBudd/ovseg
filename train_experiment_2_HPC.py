from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i")

args = parser.parse_args()

model_name = 'cubed_patches'
if int(args.i) == 0:
    val_fold = 3
    p_name = 'pod_half'
elif int(args.i) == 1:
    val_fold = 4
    p_name = 'pod_half'
elif int(args.i) == 2:
    val_fold = 3
    p_name = 'om_half'
elif int(args.i) == 3:
    val_fold = 4
    p_name = 'om_half'


model_params = get_model_params_3d_nnUNet([48, 192, 192], 2,
                                          use_prg_trn=False,
                                          fp32=True)

model = SegmentationModel(val_fold=val_fold,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()
