from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i")

args = parser.parse_args()

if int(args.i) < 2:
    model_params = get_model_params_3d_nnUNet([56, 192, 160], 2,
                                              use_prg_trn=True)
    model_name = 'prg_trn'
else:
    model_params = get_model_params_3d_nnUNet([48, 192, 192], 2,
                                              use_prg_trn=True)
    model_name = 'prg_trn_cubes'

val_fold = 5
p_name = 'pod_half' if args.i in [0, 2] else 'om_half'

model = SegmentationModel(val_fold=val_fold,
                          data_name='OV04',
                          preprocessed_name=p_name,
                          model_name=model_name,
                          model_parameters=model_params)
model.training.train()
model.eval_raw_dataset('BARTS', save_preds=False, save_plots=False)

