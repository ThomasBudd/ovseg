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
        self.np_augmentation = np_concatenated_augmentation(self.np_params)
