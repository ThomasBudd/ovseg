from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
from ovseg.model.SegmentationEnsembleV2 import SegmentationEnsembleV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_UNet
from ovseg.utils.io import load_pkl
from time import sleep
import os
from ovseg import OV_PREPROCESSED, OV_DATA_BASE

'''
Script for preprocessing and training a cascade in one go.
'''

# name of your raw dataset
data_name = 'kits21_small'
# name of lowres preprocessed data
preprocessed_name_lowres = 'MY_PREPROCESSED_NAME_lowres'
# name of fullres preprocessed data
# WARNING: If you change the lowres model the preprocessing have to be done
# again to with a new name
preprocessed_name_fullres = 'MY_PREPROCESSED_NAME_fullres'

# give each model a unique name. This way the code will be able to identify them
# both models (lowres and fullres) will have the same name and be differentiated
# by the name of preprocessed data
model_name = 'delete_me'
val_fold = 0
# %% preprocess lowres data if it hasn't been done yet
if not os.path.exists(os.path.join(OV_PREPROCESSED, data_name, preprocessed_name_lowres)):
    
    # downsample in xy plane by factor 2 for lowres model
    pooling_stride = (1,2,2)
    
    # ADD SOME PREPROCESSING PARAMETERS HERE
    prep = SegmentationPreprocessingV2(apply_resizing=True, 
                                       apply_pooling=True, 
                                       apply_windowing=True,
                                       pooling_stride=pooling_stride)
    prep.plan_preprocessing_raw_data(data_name)

    prep.preprocess_raw_data(raw_data=data_name,
                             preprocessed_name=preprocessed_name_lowres)


# %% now get hyper-parameters for low resolution and train
patch_size = [32, 256, 256]
n_2d_convs = 2
use_prg_trn = False # on low resolution prg trn can harm the performance
n_fg_classes = 1
use_fp32 = False
model_params = get_model_params_3d_UNet(patch_size=patch_size,
                                        n_2d_convs=n_2d_convs,
                                        use_prg_trn=use_prg_trn,
                                        n_fg_classes=n_fg_classes,
                                        fp32=use_fp32)

# CHANGE YOUR HYPER-PARAMETERS FOR LOWRES STAGE HERE!
model_params['training']['num_epochs'] = 100
model_params['network']['filters'] = 8
model_params['data']['n_folds'] = 2

for val_fold in range(2):
    model = SegmentationModelV2(val_fold=val_fold,
                                data_name=data_name,
                                model_name=model_name,
                                preprocessed_name=preprocessed_name_lowres,
                                model_parameters=model_params)
    # execute the trainig, simple as that!
    # It will check for previous checkpoints and load them
    model.training.train()
    
    if val_fold < model_params['data']['n_folds']:
        model.eval_validation_set()

# %% now we have to wait until all models have finished their predicitons
# we need the predictions from the low resolution before we can start the preprocessing
# of the next stage
wait = True
while wait:
    num_epochs = model_params['training']['num_epochs']
    not_finished_folds = []
    for fold in range(model_params['data']['n_folds']):
        # path to training checkpoints
        path_to_attr = os.path.join(OV_DATA_BASE,
                                    'trained_models',
                                    data_name,
                                    preprocessed_name_lowres,
                                    model_name,
                                    f'fold_{fold}',
                                    'attribute_checkpoint.pkl')
        if not os.path.exists(path_to_attr):
            print(f'No checkpoint found for fold {fold}. Training not started?')
            not_finished_folds.append(fold)
            continue

        attr = load_pkl(path_to_attr)

        if attr['epochs_done'] < attr['num_epochs']:
            not_finished_folds.append(fold)

    if len(not_finished_folds) > 0:
        print(f'Waiting for folds {not_finished_folds}')
        sleep(60)
    else:
        wait = False

# uncomment to evaluate ensemble e.g. of cross-validation models
ens = SegmentationEnsembleV2(val_fold=list(range(model_params['data']['n_folds'])),
                              model_name=model_name,
                              data_name=data_name,
                              preprocessed_name=preprocessed_name_lowres)
ens.eval_raw_dataset('kits21_small_v2')

# %% preprocess fullres data if it hasn't been done yet
if not os.path.exists(os.path.join(OV_PREPROCESSED, data_name, preprocessed_name_fullres)):
    
    prev_stage = {'data_name': data_name,
                  'preprocessed_name': preprocessed_name_lowres,
                  'model_name': model_name}
    
    # ADD SOME PREPROCESSING PARAMETERS HERE
    prep = SegmentationPreprocessingV2(apply_resizing=True, 
                                       apply_pooling=False, 
                                       apply_windowing=True,
                                       prev_stage_for_input=prev_stage)
    prep.plan_preprocessing_raw_data(data_name)

    prep.preprocess_raw_data(raw_data=data_name,
                             preprocessed_name=preprocessed_name_fullres)

# %% now get hyper-parameters for full resolution and train
patch_size = [32, 256, 256]
n_2d_convs = 2
use_prg_trn = True
n_fg_classes = 1
use_fp32 = False
out_shape = [[24, 128, 128], [24, 192, 192], [24, 256, 256], [32, 256, 256]]
model_params = get_model_params_3d_UNet(patch_size=patch_size,
                                        n_2d_convs=n_2d_convs,
                                        use_prg_trn=use_prg_trn,
                                        n_fg_classes=n_fg_classes,
                                        fp32=use_fp32,
                                        out_shape=out_shape)

# for the cascade we input the masks of the previous stage
model_params['network']['in_channels'] = 1 + n_fg_classes
model_params['training']['num_epochs'] = 100
model_params['network']['filters'] = 8
model_params['data']['n_folds'] = 2

for val_fold in range(2):
    # CHANGE YOUR HYPER-PARAMETERS FOR LOWRES STAGE HERE!
    model = SegmentationModelV2(val_fold=val_fold,
                                data_name=data_name,
                                model_name=model_name,
                                preprocessed_name=preprocessed_name_fullres,
                                model_parameters=model_params)
    
    
    # execute the trainig, simple as that!
    # It will check for previous checkpoints and load them
    model.training.train()
    
    if val_fold < model_params['data']['n_folds']:
        model.eval_validation_set()
    
# uncomment to evaluate ensemble e.g. of cross-validation models
ens = SegmentationEnsembleV2(val_fold=list(range(model_params['data']['n_folds'])),
                              model_name=model_name,
                              data_name=data_name,
                              preprocessed_name=preprocessed_name_fullres)
# typically I train all folds on different GPUs in parallel, this let's you wait
# until all trainings are done
ens.wait_until_all_folds_complete()
# evaluate ensemble on test data
ens.eval_raw_dataset('kits21_small_v2')