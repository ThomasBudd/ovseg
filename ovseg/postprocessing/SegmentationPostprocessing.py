import numpy as np
import torch
from ovseg.utils.interp_utils import resize_sample
from ovseg.utils.torch_np_utils import check_type
from skimage.measure import label


class SegmentationPostprocessing(object):

    def __init__(self, apply_small_component_removing=False,
                 volume_thresholds=None):
        self.apply_small_component_removing = apply_small_component_removing
        self.volume_thresholds = volume_thresholds

        if self.apply_small_component_removing and \
                self.volume_thresholds is None:
            raise ValueError('No volume thresholds given.')
        if not isinstance(self.volume_thresholds, (list, tuple, np.ndarray)):
            self.volume_thresholds = [self.volume_thresholds]

    def __call__(self, volume, orig_shape=None):
        return self.postprocess_volume(volume, orig_shape)

    def postprocess_volume(self, volume, orig_shape=None):
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

        # first fun step: let's reshape to original size
        # before going to hard labels
        if orig_shape is not None and orig_shape != inpt_shape:
            orig_shape = np.array(orig_shape)
            order = 3 if is_np else 1
            volume = resize_sample(volume, orig_shape, order)

            # we will need this factor when removing the small connceted
            # components. If we have twice as many voxel after resizing
            # we also only remove the component if it has twice as many
            # voxel as the threshold
            vol_threshold_fac = np.prod(orig_shape) / np.prod(inpt_shape)
        else:
            # no resizing, no change.
            vol_threshold_fac = 1

        # no we go from soft to hard labels
        if is_np:
            volume = np.argmax(volume, 0)
        else:
            volume = torch.argmax(volume, 0).cpu().detach().numpy()

        if self.apply_small_component_removing:
            # this can only be done on the CPU
            volume = self.remove_small_component(volume, vol_threshold_fac)

        return volume

    def remove_small_component(self, volume, vol_threshold_fac=1):
        if not isinstance(volume, np.ndarray):
            raise TypeError('Input must be np.ndarray')
        if not len(volume.shape) == 3:
            raise ValueError('Volume must be 3d array')

        # stores all coordinates of small components as 0 and rest as 1
        mask = np.ones_like(volume)

        num_classes = volume.max()
        if len(self.volume_thresholds) == 1:
            # if we only have one threshold we will apply it for all
            # lesion types
            thresholds = num_classes * self.volume_thresholds
        else:
            thresholds = self.volume_thresholds
            if len(self.volume_thresholds) < num_classes:
                raise ValueError('Less thresholds then fg classe given. '
                                 'Use either one threshold that is applied '
                                 'for all fg classes or')

        # we allow for different thresholds for the different lesions
        for i, tr in enumerate(thresholds):
            components = label(volume == i+1)
            n_comps = components.max()
            for j in range(1, n_comps + 1):
                comp = components == j
                if np.sum(comp) < tr * vol_threshold_fac:
                    mask[comp] = 0

        # done! The mask is 0 where all the undesired components are
        return mask * volume

    def infere_volume_thresholds(self, pred_folder, gt_folder):
        raise NotImplementedError('Inference for lesion volume not '
                                  'implemented yet.')
