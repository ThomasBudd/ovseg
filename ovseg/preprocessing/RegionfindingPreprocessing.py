from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from scipy.ndimage.morphology import binary_dilation
import numpy as np
import torch

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
        self._slow_dilation_warning_printed = False
    
    def use_masks(self):
        return True
    
    def _np_add_mask_to_volume(self, volume):
        
        if not self._slow_dilation_warning_printed:
            print('Warning! It seems like the mask for the region finding is computed on the CPU '
                  'because torch didn\'t find a GPU. The dialation is very slow on the CPU!')
            self._slow_dilation_warning_printed = True
    
        dtype = volume.dtype
        bin_lb = volume[-1] > 0
        # compute the dialated edge
        lb_dial_edge = binary_dilation(bin_lb, self.selem).astype(float) - bin_lb.astype(float)
        # now keep everything in the loss function, but the dialated edge
        mask = 1 - lb_dial_edge
        mask = mask[np.newaxis].astype(dtype)
        if super().use_masks():
            volume[-2:-1] *= mask
        else:
            volume = np.concatenate([volume[:-1], mask, volume[-1:]])
        return volume

    def _torch_add_mask_to_volume(self, volume):
        
        dtype = volume.dtype
        bin_lb_cuda = (volume[-1:] > 0).type(torch.float)
        elem_cuda = torch.from_numpy(self.selem).cuda().type(torch.float)
        padding = [(s-1)//2 for s in elem_cuda.shape]
        
        dial = torch.nn.functional.conv3d(bin_lb_cuda.unsqueeze(0),
                                          elem_cuda.unsqueeze(0).unsqueeze(0),
                                          padding=padding)[0]
        dial = (dial > 0).type(torch.float)
        dial_edge = dial - bin_lb_cuda
        mask = (1 - dial_edge).type(dtype)

        if super().use_masks():
            volume[-2:-1] *= mask
        else:
            volume = torch.cat([volume[:-1], mask, volume[-1:]])
        return volume
        
    def maybe_add_mask_to_volume(self, volume):
        if torch.is_tensor(volume):
            return self._torch_add_mask_to_volume(volume)
        else:
            return self._np_add_mask_to_volume(volume)
