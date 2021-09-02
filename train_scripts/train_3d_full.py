from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_cascade
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i")

args = parser.parse_args()

fold = int(args.i) % 5
les = 'pod' if int(args.i) < 5 else 'om'

model_params = get_model_params_3d_cascade(les + '_half',
                                           'nnUNet_benchmark',
                                           [32, 224, 224],
                                           2)

model = SegmentationModel(val_fold=fold,
                          data_name='OV04',
                          preprocessed_name=les + '_full',
                          model_name='nnUNet_cascade_benchmark',
                          model_parameters=model_params)
model.training.train()
model.eval_validation_set(save_preds=True)
model.eval_training_set()
if fold > 4:
    model.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)

ens = SegmentationEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name=les+'_full',
                           model_name='nnUNet_cascade_benchmark')

if ens.all_folds_complete():
    ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)
