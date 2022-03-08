from ovseg.augmentation.SpatialAugmentation import SpatialAugmentation
from ovseg.augmentation.GrayValueAugmentation import GrayValueAugmentation
from ovseg.augmentation.myRandAugment import torch_myRandAugment
from ovseg.augmentation.MaskAugmentation import MaskAugmentation
from ovseg.augmentation.GridAugmentation import torch_inplane_grid_augmentations
from ovseg.augmentation.GrayValueAugmentation import torch_gray_value_augmentation
import torch.nn as nn


class ConcatenatedAugmentation(object):
    '''
    SegmentationAugmentation(...)

    Performs spatial and gray value augmentations
    '''

    def __init__(self, augmentation_params):

        if augmentation_params is None:
            self.augmentation_params = {}
        else:
            self.augmentation_params = augmentation_params
        self.augmentations = []

        for key in self.augmentation_params.keys():

            if key.lower() not in ['grayvalue', 'grayvalueaugmentation',
                                   'spatial', 'spatialaugmentation',
                                   'mask', 'maskaugmentation',
                                   'medrandaugment', 'medrandaug']:
                raise ValueError('key '+str(key)+' of augmentation params'
                                 'did not match implemented augmentation '
                                 'methods.')
        # we first add spatial augmentations for efficiency
        for key in self.augmentation_params.keys():
            params = self.augmentation_params[key]
            if key in ['spatial', 'spatialaugmentation']:
                self.augmentations.append(SpatialAugmentation(**params))

        # next gray value augmenations
        for key in self.augmentation_params.keys():
            if key.lower() in ['grayvalue', 'grayvalueaugmentation']:
                params = self.augmentation_params[key]
                self.augmentations.append(GrayValueAugmentation(**params))

        # last mask augmentations
        for key in self.augmentation_params.keys():
            if key.lower() in ['mask', 'maskaugmentation']:
                params = self.augmentation_params[key]
                self.augmentations.append(MaskAugmentation(**params))

    def augment_image(self, img):
        # augment_image(img)
        for augmentation in self.augmentations:
            img = augmentation.augment_image(img)
        return img

    def augment_sample(self, sample):
        # augment_sample(sample)
        for augmentation in self.augmentations:
            sample = augmentation.augment_sample(sample)
        return sample

    def augment_batch(self, batch):
        # augment_batch(batch)
        for augmentation in self.augmentations:
            batch = augmentation.augment_batch(batch)
        return batch

    def augment_volume(self, volume, is_inverse: bool = False):
        '''
        augment_volume(volume, is_inverse=False, do_augs=None)
        volume:
            - 3d or 4d tensor or np.ndarray
        is_inverse:
            - if forward or inverse augmentation is applied for TTA
        '''

        # apply each augmentation
        for augmentation in self.augmentations:
            volume = augmentation.augment_volume(volume, is_inverse)
        return volume


# %%
class torch_concatenated_augmentation(nn.Module):

    def __init__(self, torch_params={}):

        super().__init__()

        for key in torch_params:
            assert key in ['grid_inplane', 'grayvalue', 'myRandAugment'], \
            'got unrecognised augmentation ' + key

        self.aug_list = []
        if 'grid_inplane' in torch_params:
            self.aug_list.append(torch_inplane_grid_augmentations(**torch_params['grid_inplane']))

        if 'grayvalue' in torch_params:
            self.aug_list.append(torch_gray_value_augmentation(**torch_params['grayvalue']))

        if 'myRandAugment' in torch_params:
            self.aug_list.append(torch_myRandAugment(**torch_params['myRandAugment']))

        if len(self.aug_list) > 0:
            self.module = nn.Sequential(*self.aug_list)
        else:
            self.module = nn.Identity

    def forward(self, xb):
        return self.module(xb)

    def update_prg_trn(self, param_dict, h, indx=None):

        for aug in self.aug_list:
            aug.update_prg_trn(param_dict, h, indx)


# %%
class np_concatenated_augmentation():

    def __init__(self, np_params={}):

        if 'grayvalue' in np_params.keys():
            raise NotImplementedError('gray value augmentations not implemented for np yet...')

        for key in np_params:
            assert key in ['mask'], 'got unrecognised augmentation ' + key

        self.ops_list = []
        if 'mask' in np_params:
            self.ops_list.append(MaskAugmentation(**np_params['mask']))

    def __call__(self, xb):
        for op in self.ops_list:
            xb = op(xb)
        return xb

    def update_prg_trn(self, param_dict, h, indx=None):

        for aug in self.ops_list:
            aug.update_prg_trn(param_dict, h, indx)

