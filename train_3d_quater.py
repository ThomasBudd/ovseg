from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i")

args = parser.parse_args()

model_params = get_model_params_3d_nnUNet([48, 96, 96], 1)

if int(args.i) == 0:
    les = 'pod'
elif int(args.i) == 1:
    les = 'om'

for fold in range(6):
    model = SegmentationModel(val_fold=fold,
                              data_name='OV04',
                              preprocessed_name=les+'_quater',
                              model_name=les+'_quater_benchmark',
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set(save_preds=False)
    model.eval_training_set()

    if fold == 4:
        ens = SegmentationEnsemble(val_fold=list(range(5)),
                                   data_name='OV04',
                                   preprocessed_name=les+'_quater',
                                   model_name=les+'_quater_benchmark')

        assert ens.all_folds_complete()

        ens.eval_raw_dataset('BARTS', save_preds=False, save_plots=False)
