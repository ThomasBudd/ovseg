from ovseg.augmentation.SpatialAugmentation import SpatialAugmentation
from ovseg.augmentation.GrayValueAugmentation import GrayValueAugmentation
from ovseg.augmentation.MaskAugmentation import MaskAugmentation


class SegmentationAugmentation(object):
    '''
    SegmentationAugmentation(...)

    Performs spatial and gray value augmentations
    '''

    def __init__(self, augmentation_params):

        self.augmentation_params = augmentation_params
        self.augmentations = {}

        for key in self.augmentation_params.keys():
            params = self.augmentation_params[key]
            if key == 'grayvalue':
                self.augmentations[key] = GrayValueAugmentation(**params)
            elif key == 'spatial':
                self.augmentations[key] = SpatialAugmentation(**params)
            elif key == 'mask':
                self.augmentations[key] = MaskAugmentation(**params)
            else:
                raise ValueError('key '+str(key)+' of augmentation params'
                                 'did not match implemented augmentation '
                                 'methods.')
        # the dict of all methods we want to apply in the correct order
        self.keys = [key for key in ['spatial', 'grayvalue', 'mask']
                     if key in self.augmentations]

    def augment_image(self, img):
        # augment_image(img)
        for key in self.keys:
            img = self.augmentation[key].augment_image(img)
        return img

    def augment_sample(self, sample):
        # augment_sample(sample)
        for key in self.keys:
            sample = self.augmentation[key].augment_sample(sample)
        return sample

    def augment_batch(self, batch):
        # augment_batch(batch)
        for key in self.keys:
            batch = self.augmentation[key].augment_batch(batch)
        return batch

    def augment_volume(self, volume, is_inverse: bool = False, do_augs=None):
        '''
        augment_volume(volume, is_inverse=False, do_augs=None)
        volume:
            - 3d or 4d tensor or np.ndarray
        is_inverse:
            - if forward or inverse augmentation is applied for TTA
        do_augs:
            - list of bools if each augmentation should be applied to the
              volume, default True for all
        '''
        # by default apply all augmentations
        if do_augs is None:
            do_augs = [True for _ in range(len(self.augmentations))]
        assert len(do_augs) == len(self.augmentations)
        # apply each augmentation
        for do_aug, key in zip(do_augs, self.keys):
            if do_aug:
                volume = self.augmentation[key].augment_volume(volume,
                                                               is_inverse)
        return volume
