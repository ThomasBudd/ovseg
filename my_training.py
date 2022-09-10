from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.SegmentationEnsembleV2 import SegmentationEnsembleV2
from ovseg.model.model_parameters_segmentation import get_model_params_3d_UNet

# name of your raw dataset
data_name = 'kits21_small'
# same name as in the preprocessing script
preprocessed_name = 'delete_me'
# give each model a unique name. This way the code will be able to identify them
model_name = 'delete_me'
# which fold of the training is performed?
# Example 5-fold cross-vadliation: CV folds are 0,1,...,4.
#                                  For each val_fold > 4 no CV is applied and 
#                                  100% of the training data is used
val_fold = 0

# now get hyper-parameters
# patch size used during (last stage of) training and inference
# z axis first, then xy
patch_size = [32, 256, 256]
# for standard UNet the number of inplane convolutions
n_2d_convs = 3
# wheter to use progressive learning or not. I often found it to have no
# effect on the performance, but reduces training time by up to 40%
use_prg_trn = True
# number of different foreground classes you want to segment
n_fg_classes = 1
# it is recommended to perform the training with mixed precision (fp16)
# instead of full precision (fp32)
use_fp32 = False
# shapes introduced to the network during progressive learning
# rule of thumb: reduce total number of voxels by a factor of 4,3,2 in the
# first three stages and train last stage as usual
# be careful that the patch size is still executable for your U-Net
# e.g. a U-Net that downsamples 4 times inplane should have a patch size
# where the inplane size is divisible by 2**4 
out_shape = [[24, 128, 128], [24, 192, 192], [24, 256, 256], [32, 256, 256]]


model_params = get_model_params_3d_UNet(patch_size=patch_size,
                                        n_2d_convs=n_2d_convs,
                                        use_prg_trn=use_prg_trn,
                                        n_fg_classes=n_fg_classes,
                                        fp32=use_fp32,
                                        out_shape=out_shape)


model_params['training']['num_epochs'] = 100
model_params['network']['filters'] = 9
model_params['data']['n_folds'] = 2
# CHANGE YOUR HYPER-PARAMETERS HERE! For example
# change batch size to 4
#model_params['data']['trn_dl_params']['batch_size'] = 4
#model_params['data']['val_dl_params']['batch_size'] = 4
# change momentum
#model_params['training']['opt_params']['momentum'] = 0.98
# change weight decay
#model_params['training']['opt_params']['weight_decay'] = wd

# creat model object.
# this object holds all objects that define a deep neural network model
#   - preprocessing
#   - augmentation
#   - training
#   - slinding window evaluation
#   - postprocessing
#   - data and data sampling
#   - functions to iterate over datasets
#   - I'm sure I forgot something

for val_fold in range(3):
    model = SegmentationModelV2(val_fold=val_fold,
                                data_name=data_name,
                                model_name=model_name,
                                preprocessed_name=preprocessed_name,
                                model_parameters=model_params)
    # execute the trainig, simple as that!
    # It will check for previous checkpoints and load them
    model.training.train()
    
    # if cross-validation is applied you can evaluate the validation scans like this
    # as stated above, val_fold > n_folds means using 100% training data e.g. no validation data
    if val_fold < model_params['data']['n_folds']:
        model.eval_validation_set()

# uncomment to evaluate raw (test) dataset with the model
model.eval_raw_dataset('kits21_small')


# uncomment to evaluate ensemble e.g. of cross-validation models
ens = SegmentationEnsembleV2(val_fold=list(range(model_params['data']['n_folds'])),
                              model_name=model_name,
                              data_name=data_name,
                              preprocessed_name=preprocessed_name)
# typically I train all folds on different GPUs in parallel, this let's you wait
# until all trainings are done
ens.wait_until_all_folds_complete()
#evaluate ensemble on test data
ens.eval_raw_dataset('kits21_small_v2')


