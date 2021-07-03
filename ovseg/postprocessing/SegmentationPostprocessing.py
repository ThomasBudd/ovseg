import numpy as np
import torch
from ovseg.utils.torch_np_utils import check_type
from skimage.measure import label
from skimage.transform import resize
from torch.nn.functional import interpolate


class SegmentationPostprocessing(object):

    def __init__(self, apply_small_component_removing=False,
                 volume_thresholds=None,
                 mask_with_bin_pred=False):
        self.apply_small_component_removing = apply_small_component_removing
        self.volume_thresholds = volume_thresholds
        self.mask_with_bin_pred=mask_with_bin_pred

        if self.apply_small_component_removing and \
                self.volume_thresholds is None:
            raise ValueError('No volume thresholds given.')
        if not isinstance(self.volume_thresholds, (list, tuple, np.ndarray)):
            self.volume_thresholds = [self.volume_thresholds]

    def postprocess_volume(self, volume, bin_pred=None, spacing=None, orig_shape=None):
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
        if self.mask_with_bin_pred:
            if bin_pred is None:
                raise ValueError('Trying to multiply the prediction with the bin_pred of the '
                                 'previous stages, but no such array was given.')

            if len(bin_pred.shape) == 3:
                if isinstance(bin_pred, np.ndarray):
                    bin_pred = bin_pred[np.newaxis]
                else:
                    bin_pred = bin_pred.unsqueeze(0)
        # first fun step: let's reshape to original size
        # before going to hard labels
        if orig_shape is not None:
            if np.any(orig_shape != inpt_shape):
                orig_shape = np.array(orig_shape)
                if torch.cuda.is_available():
                    with torch.no_grad():
                        if is_np:
                            volume = torch.from_numpy(volume).to(self.dev).type(torch.float)
                        size = [int(s) for s in orig_shape]
                        volume = interpolate(volume.unsqueeze(0),
                                             size=size,
                                             mode='trilinear')[0]
                        if self.mask_with_bin_pred:
                            if isinstance(bin_pred, np.ndarray):
                                bin_pred = torch.from_numpy(bin_pred).to(self.dev).type(torch.float)
                            bin_pred = interpolate(bin_pred.unsqueeze(0),
                                                   size=size,
                                                   mode='nearest')[0, 0]
                else:
                    if not is_np:
                        volume = volume.cpu().numpy()
                    volume = np.stack([resize(volume[c], orig_shape, 1)
                                       for c in range(volume.shape[0])])
                    if self.mask_with_bin_pred:
                        if torch.is_tensor(bin_pred):
                            bin_pred = bin_pred.cpu().numpy()
                        bin_pred = np.stack([resize(bin_pred[c], orig_shape, 0)
                                           for c in range(bin_pred.shape[0])])

        # now change from soft to hard labels 
        if torch.is_tensor(volume):
            volume = volume.cpu().numpy()
        if self.mask_with_bin_pred:
            if torch.is_tensor(bin_pred):
                bin_pred = bin_pred.cpu().numpy()

        # now change from soft to hard labels and multiply by the binary prediction
        volume = np.argmax(volume, 0)
        if self.mask_with_bin_pred:
            # now we're finally doing what we're asking the whole time about!
            volume *= bin_pred[0]


        if self.apply_small_component_removing:
            # this can only be done on the CPU
            volume = self.remove_small_components(volume, spacing)

        return volume.astype(np.uint8)

    def postprocess_data_tpl(self, data_tpl, prediction_key, bin_pred=None):

        pred = data_tpl[prediction_key]

        if 'orig_shape' in data_tpl:
            # the data_tpl has preprocessed data.
            # predictions in both preprocessed and original shape will be added
            data_tpl[prediction_key] = self.postprocess_volume(pred,
                                                               bin_pred=bin_pred,
                                                               spacing=data_tpl['spacing'],
                                                               orig_shape=None)
            spacing = data_tpl['orig_spacing'] if 'orig_spacing' in data_tpl else None
            shape = data_tpl['orig_shape']
            data_tpl[prediction_key+'_orig_shape'] = self.postprocess_volume(pred,
                                                                             bin_pred=bin_pred,
                                                                             spacing=spacing,
                                                                             orig_shape=shape)
        else:
            # in this case the data is not preprocessed
            orig_shape = data_tpl['image'].shape
            data_tpl[prediction_key] = self.postprocess_volume(pred,
                                                               bin_pred=bin_pred,
                                                               spacing=data_tpl['spacing'],
                                                               orig_shape=orig_shape)
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
