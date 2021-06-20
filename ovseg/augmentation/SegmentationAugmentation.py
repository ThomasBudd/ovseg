from ovseg.augmentation.ConcatenatedAugmentation import torch_concatenated_augmentation, \
    np_concatenated_augmentation


class SegmentationAugmentation(object):
    '''
    SegmentationAugmentation(...)

    Performs spatial and gray value augmentations
    '''

    def __init__(self, torch_params={}, np_params={}):

        self.torch_params = torch_params
        self.np_params = np_params

        self.torch_augmentation = torch_concatenated_augmentation(self.torch_params)
        if self.np_params == {}:
            self.np_augmentation = None
        else:
            self.np_augmentation = np_concatenated_augmentation(self.np_params)

    def update_prg_trn(self, param_dict, h):

        self.torch_augmentation.update_prg_trn(param_dict, h)
        if self.np_augmentation is not None:
            self.np_augmentation.update_prg_trn(param_dict, h)
