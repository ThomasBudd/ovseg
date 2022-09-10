from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
from ovseg.model.SegmentationEnsembleV2 import SegmentationEnsembleV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_UNet
from ovseg.utils.io import load_pkl
from time import sleep
import os
from ovseg import OV_PREPROCESSED, OV_DATA_BASE

'''
Script for preprocessing and training of kidney masses using deep supervision
e.g. only training and validating where the previous stage (kidney model) found foreground
'''

# name of your raw dataset
data_name = 'kits21_small'
# name of preprocessed data
preprocessed_name = 'kidneys_masses'

# give each model a unique name. This way the code will be able to identify them
# both models (lowres and fullres) will have the same name and be differentiated
# by the name of preprocessed data

model_name = 'delete_me'
val_fold = 0
# %% preprocess lowres data if it hasn't been done yet
if not os.path.exists(os.path.join(OV_PREPROCESSED, data_name, preprocessed_name)):
    
    # doing a cascade inputting the previous stage
    prev_stage = {'data_name': data_name,
                  'preprocessed_name': 'kidneys_fullres',
                  'model_name': 'U-Net32'}
    
    # ADD SOME PREPROCESSING PARAMETERS HERE
    prep = SegmentationPreprocessingV2(apply_resizing=True, 
                                       apply_pooling=False, 
                                       apply_windowing=True,
                                       target_spacing=[2,1,1], # downsample to 4mm^3
                                       lb_classes=[2,3], # exclude class 1 (kidneys) and only segment 2 (tumors) and 3 (cysts)
                                       prev_stage_for_mask=prev_stage) 
    prep.plan_preprocessing_raw_data(data_name)

    prep.preprocess_raw_data(raw_data=data_name,
                             preprocessed_name=preprocessed_name)


# %% now get hyper-parameters for low resolution and train
patch_size = [48, 96, 96]
n_2d_convs = 1
use_prg_trn = True # on low resolution prg trn can harm the performance
n_fg_classes = 2
use_fp32 = False
out_shapes = [[24, 64, 64], [32, 64, 64], [32, 80, 80], [48, 96, 96]]
model_params = get_model_params_3d_UNet(patch_size=patch_size,
                                        n_2d_convs=n_2d_convs,
                                        use_prg_trn=use_prg_trn,
                                        n_fg_classes=n_fg_classes,
                                        fp32=use_fp32,
                                        out_shape=out_shapes)

# tell data object it should also load the masks
model_params['data']['folders'] = ['images', 'labels', 'masks']
model_params['data']['keys'] = ['image', 'label', 'mask']

for s in ['trn_dl_params', 'val_dl_params']:
    # tell dataloaders to use the bias where first a class and then a
    # foreground voxel is chosen
    model_params['data'][s]['bias'] = 'cl_fg'
    # number of foreground classes
    model_params['data'][s]['n_fg_classes'] = 2

# tell the training object to use the masks for the loss function
model_params['training']['batches_have_masks'] = True
# apply the mask during post-processing
model_params['postprocessing'] = {'mask_with_reg': True}
model_params['training']['num_epochs'] = 100
model_params['network']['filters'] = 8
model_params['data']['n_folds'] = 2

for val_fold in range(2):
    model = SegmentationModelV2(val_fold=val_fold,
                                data_name=data_name,
                                model_name=model_name,
                                preprocessed_name=preprocessed_name,
                                model_parameters=model_params)
    
    # execute the trainig, simple as that!
    # It will check for previous checkpoints and load them
    model.training.train()
    
    if val_fold < model_params['data']['n_folds']:
        model.eval_validation_set()
        
ens = SegmentationEnsembleV2(val_fold=list(range(model_params['data']['n_folds'])),
                             model_name=model_name,
                             data_name=data_name,
                             preprocessed_name=preprocessed_name)
# typically I train all folds on different GPUs in parallel, this let's you wait
# until all trainings are done
ens.wait_until_all_folds_complete()
# evaluate ensemble on test data
ens.eval_raw_dataset('kits21_small_v2')