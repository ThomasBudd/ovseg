from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
from ovseg.model.SegmentationEnsembleV2 import SegmentationEnsembleV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_UNet
from ovseg.utils.io import load_pkl
from time import sleep
import os
from ovseg import OV_PREPROCESSED, OV_DATA_BASE

'''
Script for preprocessing and training lowres kidney models
'''

# name of your raw dataset
data_name = 'kits21'
# name of preprocessed data
preprocessed_name = 'kidneys_lowres'

# give each model a unique name. This way the code will be able to identify them
# both models (lowres and fullres) will have the same name and be differentiated
# by the name of preprocessed data

model_name = 'U-Net32'
val_fold = 0
# %% preprocess lowres data if it hasn't been done yet
if not os.path.exists(os.path.join(OV_PREPROCESSED, data_name, preprocessed_name)):
    
    # ADD SOME PREPROCESSING PARAMETERS HERE
    prep = SegmentationPreprocessingV2(apply_resizing=True, 
                                       apply_pooling=False, 
                                       apply_windowing=True,
                                       target_spacing=[4,4,4], # downsample to 4mm^3
                                       reduce_lb_to_single_class=True) # in this first stage segment kidneys plus masses
    prep.plan_preprocessing_raw_data(data_name)

    prep.preprocess_raw_data(raw_data=data_name,
                             preprocessed_name=preprocessed_name)


# %% now get hyper-parameters for low resolution and train
patch_size = [64, 64, 64]
n_2d_convs = 0
use_prg_trn = False # on low resolution prg trn can harm the performance
n_fg_classes = 1
use_fp32 = False
model_params = get_model_params_3d_UNet(patch_size=patch_size,
                                        n_2d_convs=n_2d_convs,
                                        use_prg_trn=use_prg_trn,
                                        n_fg_classes=n_fg_classes,
                                        fp32=use_fp32)

# CHANGE YOUR HYPER-PARAMETERS FOR LOWRES STAGE HERE!

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