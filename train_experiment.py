from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i")

args = parser.parse_args()

model_params = get_model_params_3d_nnUNet([56, 192, 160], 2,
                                          use_prg_trn=True)
model_name = 'prg_trn'
val_fold = int(args.i)

for p_name in ['pod_half', 'om_half']:
    model = SegmentationModel(val_fold=val_fold,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name=model_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set()
    #model.eval_raw_dataset('BARTS', save_preds=False, save_plots=False)
    if val_fold == 0:
        ens = SegmentationEnsemble(val_fold=list(range(5)),
                                   data_name='OV04',
                                   preprocessed_name=p_name,
                                   model_name='nnUNet_benchmark')
    
        if ens.all_folds_complete():
            ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)
