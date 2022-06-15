from ovseg.augmentation.myRandAugment import torch_myRandAugment
from ovseg.augmentation.MaskAugmentation import MaskAugmentation
from ovseg.augmentation.GridAugmentation import torch_inplane_grid_augmentations
from ovseg.augmentation.GrayValueAugmentation import torch_gray_value_augmentation
import torch.nn as nn


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

