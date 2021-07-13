from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from scipy.ndimage.morphology import binary_dilation
import numpy as np

class RegionfindingPreprocessing(SegmentationPreprocessing):

    def __init__(self, *args, mask_dist, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.mask_dist = mask_dist

        axes = [np.linspace(-1, 1, 2*m+1) for m in self.mask_dist]
        
        # define the ball with radius as in mask_dist
        self.selem = np.sum(np.stack(np.meshgrid(*axes, indexing='ij'))**2, 0) <= 1
        
        
        self.preprocessing_parameters = ['apply_resizing',
                                         'apply_pooling',
                                         'apply_windowing',
                                         'target_spacing',
                                         'pooling_stride',
                                         'window',
                                         'scaling',
                                         'lb_classes',
                                         'reduce_lb_to_single_class',
                                         'lb_min_vol',
                                         'n_im_channels',
                                         'do_nn_img_interp',
                                         'save_only_fg_scans',
                                         'prev_stages',
                                         'dataset_properties',
                                         'mask_dist']
    def use_masks(self):
        return True
    
    def get_mask_from_data_tpl(self, data_tpl):
        
        dtype = data_tpl['label'].dtype
        bin_lb = data_tpl['label'] > 0
        # compute the dialated edge
        lb_dial_edge = binary_dilation(bin_lb, self.selem).astype(dtype) - bin_lb.astype(dtype)
        # now keep everything in the loss function, but the dialated edge
        mask = 1 - lb_dial_edge
        return mask