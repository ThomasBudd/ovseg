from ovseg.augmentation.SpatialAugmentation import SpatialAugmentation
from ovseg.augmentation.GrayValueAugmentation import GrayValueAugmentation


class SegmentationAugmentation(object):
    '''
    SegmentationAugmentation(...)

    Performs spatial and gray value augmentations
    '''

    def __init__(self, augmentation_params):

        self.augmentation_params = augmentation_params
        self.augmentations = []
        for key in self.augmentation_params.keys():
            params = self.augmentation_params[key]
            if key in ['grayvalue', 'grayvalueaugmentation']:
                self.augmentations.append(GrayValueAugmentation(**params))
            elif key in ['spatial', 'spatialaugmentation']:
                self.augmentations.append(SpatialAugmentation(**params))
            else:
                raise ValueError('key '+str(key)+' of augmentation params'
                                 'did not match implemented augmentation '
                                 'methods.')

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
        for do_aug, augmentation in zip(do_augs, self.augmentations):
            if do_aug:
                volume = augmentation.augment_volume(volume, is_inverse)
        return volume
