from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i")

args = parser.parse_args()

model_params = get_model_params_3d_nnUNet([56, 192, 160], 2)

if int(args.i) == 0:
    folds = [0]
    p_names = ['pod_half']
elif int(args.i) == 1:
    folds = [1]
    p_names = ['pod_half']
elif int(args.i) == 2:
    folds = [2]
    p_names = ['pod_half']
elif int(args.i) == 3:
    folds = [3]
    p_names = ['pod_half']
elif int(args.i) == 4:
    folds = [4]
    p_names = ['pod_half']
elif int(args.i) == 5:
    folds = [0, 3]
    p_names = ['om_half', 'om_half']
elif int(args.i) == 6:
    folds = [1, 4]
    p_names = ['om_half', 'om_half']
elif int(args.i) == 7:
    folds = [2, 5]
    p_names = ['om_half', 'om_half']
elif int(args.i) == 8:
    folds = [5, 6, 6]
    p_names = ['pod_half', 'pod_half', 'om_half']


for fold, p_name in zip(folds, p_names):
    model = SegmentationModel(val_fold=fold,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name='nnUNet_benchmark',
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set(save_preds=False)
    model.preprocess_prediction_for_next_stage('pod_full')
    model.eval_training_set()
    if fold > 4:
        model.eval_raw_dataset('BARTS', save_preds=False, save_plots=False)

    ens = SegmentationEnsemble(val_fold=list(range(5)),
                               data_name='OV04',
                               preprocessed_name=p_name,
                               model_name=p_name+'_benchmark')

    if ens.all_folds_complete():
        ens.eval_raw_dataset('BARTS', save_preds=False, save_plots=False)
