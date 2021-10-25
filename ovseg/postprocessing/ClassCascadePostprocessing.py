import numpy as np
import torch
from ovseg.postprocessing.SegmentationPostprocessing import SegmentationPostprocessing
from skimage.transform import resize
from torch.nn.functional import interpolate
from ovseg.utils.torch_np_utils import check_type, maybe_add_channel_dim


class ClassCascadePostprocessing(SegmentationPostprocessing):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not self.mask_with_reg:
            print('mask_with_reg was set to False, but should be True. Changing this now...')
            self.mask_with_reg = True

    def postprocess_volume(self, volume, prev_pred=None, spacing=None, orig_shape=None):
        '''
        postprocess_volume(volume, orig_shape=None)

        Applies the following for post processing:
            - resizing to original voxel spacing (if given)
            - applying argmax to go from hard to soft labels
            - removing small connected components (if set to true)

        Parameters
        ----------
        volume : array tensor
            volume with soft segmentation/ output of the CNN
        orig_shape : len 3, optional
            if out_shape is given the volume is resized to original shape
            before any other postprocessing is done

        Returns
        -------
        postprocessed hard segmentation labels

        '''

        # first let's check if the input is right
        is_np, _ = check_type(volume)
        inpt_shape = np.array(volume.shape)
        if len(inpt_shape) != 4:
            raise ValueError('Expected 4d volume of shape '
                             '[n_channels, nx, ny, nz].')
        if prev_pred is None:
            raise ValueError('Previous prediction must be given as an input to '
                             'the postprocessing.')

        prev_pred = maybe_add_channel_dim(prev_pred)
        # first fun step: let's reshape to original size
        # before going to hard labels        
        if orig_shape is not None:
            if np.any(orig_shape != inpt_shape):
                orig_shape = np.array(orig_shape)
                if torch.cuda.is_available():
                    # if cuda is available we use pytorch and the GPU to do
                    # the resizing --> way faster
                    with torch.no_grad():
                        if is_np:
                            volume = torch.from_numpy(volume).to('cuda').type(torch.float)
                        size = [int(s) for s in orig_shape]
                        volume = interpolate(volume.unsqueeze(0),
                                             size=size,
                                             mode='trilinear')[0]
                        if isinstance(prev_pred, np.ndarray):
                            prev_pred = torch.from_numpy(prev_pred).to('cuda').type(torch.float)
                        prev_pred = interpolate(prev_pred.unsqueeze(0),
                                               size=size,
                                               mode='nearest')[0]
                else:
                    # otherwise we have to do it with skimage
                    # --> way slower (buuuhhh!)
                    if not is_np:
                        volume = volume.cpu().numpy()
                    volume = np.stack([resize(volume[c], orig_shape, 1)
                                       for c in range(volume.shape[0])])
                    if torch.is_tensor(prev_pred):
                        prev_pred = prev_pred.cpu().numpy()
                    prev_pred = np.stack([resize(prev_pred[c], orig_shape, 0)
                                       for c in range(prev_pred.shape[0])])

        # now mask and change from soft to hard labels 
        if torch.is_tensor(volume):
            volume = torch.argmax(volume, 0).type(torch.float)
            mask = (prev_pred == 0).type(torch.float)
            volume = volume * mask
            volume = volume.cpu().numpy()
        else:
            volume = np.argmax(volume, 0).astype(float)
            mask = (prev_pred == 0).astype(float)
            volume = volume * mask
        # the volume is 3d now

        if self.apply_small_component_removing:
            # this can only be done on the CPU
            volume = self.remove_small_components(volume, spacing)

        volume = volume.astype(np.uint8)

        if self.lb_classes is not None:
            # now let's convert back from interger encoding to the classes
            volume_lb = np.zeros_like(volume)
            for i, c in enumerate(self.lb_classes):
                volume_lb[volume == i+1] = c
            volume = volume_lb
        
        # now in the end we fill in the previous prediction again
        if torch.is_tensor(prev_pred):
            prev_pred.cpu().numpy()
        
        # the volume should be 0 where prev_pred == 0, this expression
        # is typically faster then 
        # volume[prev_pred > 0] = prev_prev[prev_pred> 0]
        volume += prev_pred[0]

        return volume