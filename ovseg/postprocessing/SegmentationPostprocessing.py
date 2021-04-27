import numpy as np
import torch
from ovseg.utils.interp_utils import resize_sample
from ovseg.utils.torch_np_utils import check_type
from skimage.measure import label
from skimage.transform import resize
from torch.nn.functional import interpolate


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

    def postprocess_volume(self, volume, spacing=None, orig_shape=None, had_z_first=False):
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
        if orig_shape is not None:
            if np.any(orig_shape != inpt_shape):
                orig_shape = np.array(orig_shape)
                if is_np:
                    volume = np.stack([resize(volume[c], orig_shape, 3)
                                       for c in range(volume.shape[0])])
                else:
                    size = [int(s) for s in orig_shape]
                    volume = interpolate(volume.unsqueeze(0),
                                         size=size,
                                         mode='trilinear')[0]

        # now change from soft to hard labels
        if is_np:
            volume = np.argmax(volume, 0)
        else:
            volume = torch.argmax(volume, 0).cpu().detach().numpy()

        if self.apply_small_component_removing:
            # this can only be done on the CPU
            volume = self.remove_small_components(volume, spacing)

        if had_z_first:
            volume = np.stack([volume[z] for z in range(volume.shape[0])], -1)

        return volume.astype(np.uint8)

    def postprocess_data_tpl(self, data_tpl, prediction_key):

        pred = data_tpl[prediction_key]
        if 'had_z_first' in data_tpl:
            had_z_first = data_tpl['had_z_first']
        else:
            had_z_first = False

        if 'orig_shape' in data_tpl:
            # the data_tpl has preprocessed data.
            # predictions in both preprocessed and original shape will be added
            data_tpl[prediction_key] = self.postprocess_volume(pred,
                                                               spacing=data_tpl['spacing'],
                                                               orig_shape=None,
                                                               had_z_first=had_z_first)
            spacing = data_tpl['orig_spacing'] if 'orig_spacing' in data_tpl else None
            shape = data_tpl['orig_shape']
            data_tpl[prediction_key+'_orig_shape'] = self.postprocess_volume(pred,
                                                                             spacing=spacing,
                                                                             orig_shape=shape,
                                                                             had_z_first=had_z_first)
        else:
            # in this case the data is not preprocessed
            orig_shape = data_tpl['image'].shape
            data_tpl[prediction_key] = self.postprocess_volume(pred,
                                                               spacing=data_tpl['spacing'],
                                                               orig_shape=orig_shape,
                                                               had_z_first=had_z_first)
        return data_tpl

    def remove_small_components(self, volume, spacing):
        if not isinstance(volume, np.ndarray):
            raise TypeError('Input must be np.ndarray')
        if not len(volume.shape) == 3:
            raise ValueError('Volume must be 3d array')

        if not isinstance(np.ndarray, spacing):
            raise ValueError('Spacing must be a list of length 3 to represent the spatial length '
                             'of the voxel')

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
        voxel_size = np.prod(spacing)
        for i, tr in enumerate(thresholds):
            components = label(volume == i+1)
            n_comps = components.max()
            for j in range(1, n_comps + 1):
                comp = components == j
                if np.sum(comp) < tr * voxel_size:
                    mask[comp] = 0

        # done! The mask is 0 where all the undesired components are
        return mask * volume

    def infere_volume_thresholds(self, pred_folder, gt_folder):
        raise NotImplementedError('Inference for lesion volume not '
                                  'implemented yet.')
