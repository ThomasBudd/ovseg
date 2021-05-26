from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int)
# parser.add_argument("rep", type=int)
args = parser.parse_args()

for val_fold_list in [list(range(5, 8)), list(range(5))]:
    p_name = 'pod_2d'
    
    model_name = 'var_small_model_{}fCV_{}'.format(len(val_fold_list), args.gpu)
    model_params = get_model_params_2d_segmentation()
    del model_params['augmentation']['torch_params']['grayvalue']
    model_params['augmentation']['torch_params']['grid_inplane']['p_rot'] = 1.0
    model_params['augmentation']['torch_params']['grid_inplane']['p_zoom'] = 1.0
    model_params['augmentation']['torch_params']['grid_inplane']['mm_rot'] = [-15, 15]
    model_params['augmentation']['torch_params']['grid_inplane']['mm_zoom'] = [0.8, 1.2]
    model_params['training']['lr_schedule'] = 'lin_ascent_cos_decay'
    model_params['training']['lr_params'] = {'n_warmup_epochs': 50, 'lr_max': 0.02}
    model_params['network']['filters'] = 8
    model_params['network']['filters_max'] = 80
    model_params['data']['trn_dl_params']['batch_size'] = 4
    model_params['data']['val_dl_params']['batch_size'] = 4
    
    for val_fold in val_fold_list:
        model = SegmentationModel(val_fold=val_fold,
                                  data_name='OV04',
                                  preprocessed_name=p_name,
                                  model_name=model_name,
                                  model_parameters=model_params)
        model.training.train()
        model.eval_validation_set()
        model.clean()
    
    ens = SegmentationEnsemble(val_fold=val_fold_list,
                               data_name='OV04',
                               preprocessed_name=p_name,
                               model_name=model_name)
    if ens.all_folds_complete():
        ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)
    else:
        raise FileNotFoundError('ERROR: Something went wrong, why are the folds not finished yet?')
