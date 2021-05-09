from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i")

args = parser.parse_args()

if int(args.i) < 5:
    model_name = 'prg_trn_1200'
    p_name = 'pod_half'
    num_epochs = 1200
elif int(args.i) < 10:
    model_name = 'prg_trn_1400'
    p_name = 'pod_half'
    num_epochs = 1400
elif int(args.i) < 15:
    model_name = 'prg_trn_1200'
    p_name = 'om_half'
    num_epochs = 1200
elif int(args.i) < 20:
    model_name = 'prg_trn_1400'
    p_name = 'om_half'
    num_epochs = 1400


val_fold = int(args.i) % 5
model_params = get_model_params_3d_nnUNet([48, 192, 192], 2,
                                          use_prg_trn=True,
                                          fp32=True)
model_params['training']['num_epochs'] = num_epochs

model = SegmentationModel(val_fold=val_fold,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set()
