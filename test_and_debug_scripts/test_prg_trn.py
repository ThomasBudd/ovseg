from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_nnUNet


model_params = get_model_params_3d_nnUNet([48, 96, 96], 1)


prg_trn_sizes = [[48, 48, 48], [48, 64, 64], [48, 80, 80], [48, 96, 96]]

fold = 6

for les in ['pod', 'om']:
    model_params['training']['prg_trn_sizes'] = None
    model = SegmentationModel(val_fold=fold,
                              data_name='OV04',
                              preprocessed_name=les+'_quater',
                              model_name='train_default',
                              model_parameters=model_params)
    model.training.train()
    model.eval_training_set()
    model.eval_raw_dataset('BARTS')

    model_params['training']['prg_trn_sizes'] = prg_trn_sizes
    model = SegmentationModel(val_fold=fold,
                              data_name='OV04_test',
                              preprocessed_name=les+'_quater',
                              model_name='test_prg_trn',
                              model_parameters=model_params)
    model.training.train()
    model.eval_training_set()
    model.eval_raw_dataset('BARTS')
