from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
from ovseg import OV_PREPROCESSED
import os

# name of your raw dataset
data_name = 'kits21_small'
# name after preprocessing
preprocessed_name = 'delte_me'

# whether to apply resizing, pooling and windowing during preprocessing
apply_resizing = True
apply_pooing = False
apply_windowing = True

# if apply_resizing all scans are resized to this spacing (before potential pooling)
# default: inferre median voxel spacing from dataset and use this
target_spacing = None

# stride used for mean pooling of images/max pooling of labels e.g. (1,2,2)
# will only complain about pooling_stride = None if apply_pooling
pooling_stride = None

# clipping of gray values e.g. (-150, 250) for abdominal CT
# default: inferre 0.5 and 99.5 gray value of foreground voxel
window = None

# gray value scaling applied after pooling e.g.
# default: inferre Z normalization from data (recommended)
scaling = None

# if you have many classes in your segmentation problem and you only want to
# segment some of them at a time, use e.g. lb_classes = [1, 3, 4]
# default: use all classes
lb_classes = None

# removed class information and reduces segmentaiton problem to be binary
reduce_lb_to_single_class = True

# number of image channels, only tested for n_im_channels=1!!
n_im_channels = 1

# if true saves only scans that contain at least one foreground voxel
save_only_fg_scans = False

# set this variable if you want to input the segmentaiton masks from a previous
# model, e.g. in a cascade
# prev_stage_for_input = {'data_name': MY_DATA, 'preprocessed_name': MY_PREPROCESSED_NAME, 'model_name': MY_MODEL_NAME}
prev_stage_for_input = {}

# similarly to mask the segmentation, e.g. in deep supervision when segmenting first and organ and then lesions
prev_stage_for_mask = {}

# if you want to increase the size of the mask you can dialate it as a preprocessing setp
r_dial_mask = 0


# creat preprocessing object
prep = SegmentationPreprocessingV2(apply_resizing=apply_resizing,
                                   apply_pooling=apply_pooing,
                                   apply_windowing=apply_windowing,
                                   target_spacing=target_spacing,
                                   pooling_stride=pooling_stride,
                                   window=window,
                                   scaling=scaling,
                                   lb_classes=lb_classes,
                                   reduce_lb_to_single_class=reduce_lb_to_single_class,
                                   n_im_channels=n_im_channels,
                                   save_only_fg_scans=save_only_fg_scans,
                                   prev_stage_for_input=prev_stage_for_input,
                                   prev_stage_for_mask=prev_stage_for_mask,
                                   r_dial_mask=r_dial_mask)

# inferre preprocessing parameters (window, target_spacing, scaling)
prep.plan_preprocessing_raw_data(data_name)

# execute preprocessing
prep.preprocess_raw_data(data_name, preprocessed_name)

print('Preprocessing done!')