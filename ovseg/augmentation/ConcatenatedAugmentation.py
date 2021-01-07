from ovseg.augmentation.SpatialAugmentation import SpatialAugmentation
from ovseg.augmentation.GrayValueAugmentation import GrayValueAugmentation
from ovseg.augmentation.MaskAugmentation import MaskAugmentation


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
                                   'mask', 'maskaugmentation']:
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
