from ovseg.augmentation.ConcatenatedAugmentation import ConcatenatedAugmentation


class SegmentationAugmentation(object):
    '''
    SegmentationAugmentation(...)

    Performs spatial and gray value augmentations
    '''

    def __init__(self, GPU_params=None, CPU_params=None, TTA_params=None):

        self.GPU_params = GPU_params
        self.CPU_params = CPU_params
        self.TTA_params = TTA_params

        if self.GPU_params is not None:
            self.GPU_augmentation = ConcatenatedAugmentation(self.GPU_params)
        else:
            self.GPU_augmentation = None

        if self.CPU_params is not None:
            self.CPU_augmentation = ConcatenatedAugmentation(self.CPU_params)
        else:
            self.CPU_augmentation = None

        if self.TTA_params is not None:
            self.TTA = ConcatenatedAugmentation(self.TTA_params)
        else:
            self.TTA = None
