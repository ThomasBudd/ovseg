from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble

model_params = get_model_params_3d_nnUNet([56, 192, 160], 2)

p_name = 'pod_half'


for fold in range(5):
    model = SegmentationModel(val_fold=fold,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name=p_name+'_benchmark',
                              model_parameters=model_params)
    model.save_model_parameters()
    model.eval_validation_set(save_preds=False)
    model.eval_training_set()

ens = SegmentationEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name=p_name,
                           model_name=p_name+'_benchmark')

if ens.all_folds_complete():
    ens.eval_raw_dataset('BARTS', save_preds=False, save_plots=False)
