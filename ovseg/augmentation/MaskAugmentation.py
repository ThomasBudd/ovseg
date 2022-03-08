import numpy as np
import torch
from ovseg.utils.torch_np_utils import check_type, stack
try:
    from scipy.ndimage import morphology
    from skimage.measure import label
except ImportError:
    print('Caught Import Error while importing some function from scipy or skimage. '
          'Please use a newer version of gcc.')


TORCH_WARNING_PRINTED = False


class MaskAugmentation(object):
    '''
    MaksAugmentation(PARAMETERS!!)
    Performs the following augmentations:
        - morphological changes of segmentation masks
        - removing of small lesions

    Parameter:
    ----------------
    p_xxx :
        - probability with which xxx is applied to the image
    xxx_mm :
        - min and max of the uniform distribution which is used to draw the
          parameters for xxx
    vol_percentage_removal/vol_threshold_removal:
        if vol_threshold_removal and spacing is given the leions removal
        threshold is computed in real world units, else the threshold is
        computed as the percentage of the patch size
    '''

    def __init__(self, spacing=None, p_morph=0.4, radius_mm=[1, 8], p_removal=0.2,
                 vol_percentage_removal=0.15, vol_threshold_removal=None,
                 threeD_morph_ops=False, aug_channels=[1]):

        # morphological operations
        self.p_morph = p_morph
        self.radius_mm = radius_mm
        self.threeD_morph_ops = threeD_morph_ops

        # removal of small components
        self.p_removal = p_removal
        self.vol_threshold_removal = vol_threshold_removal
        self.vol_percentage_removal = vol_percentage_removal
        self.spacing = spacing

        # determins which channels are being augmented
        self.aug_channels = aug_channels

        self.morph_operations = [morphology.binary_closing,
                                 morphology.binary_dilation,
                                 morphology.binary_opening,
                                 morphology.binary_erosion]

        if spacing is not None:
            self.spacing = np.array(spacing)

    def _morphological_augmentation(self, img):

        # should be 2 or 3
        img_dim = len(img.shape)
        assert img_dim in [2, 3]

        if img_dim == 3:
            spacing = self.spacing if self.spacing is not None else \
                np.mean(img.shape) / np.array(img.shape)
            if img.shape[0] * 2 < np.min(img.shape[1:]):
                spacing = spacing[1:]
        else:
            spacing = self.spacing[1:] if self.spacing is not None else np.array([1, 1])

        classes = list(range(1, int(img.max())+1))
        # turn integer in one hot encoding boolean
        img_one_hot = np.stack([img == c for c in classes])

        # perform the operations on a random order of the classes
        np.random.shuffle(classes)

        # radius in mm, e.g. real world units
        r_mm = np.random.uniform(self.radius_mm[0], self.radius_mm[1])
        # radius in amount of pixel
        r_pixel = (r_mm / spacing).astype(int)

        # zero centered axes in mm
        axes = [np.linspace(-1 * sp * rp, sp * rp, 2 * rp + 1) for sp, rp in zip(spacing, r_pixel)]
        grid = np.stack(np.meshgrid(*axes, indexing='ij'))

        # the structure is a L2 ball with radius r_mm
        structure = np.sum(grid**2, 0) < r_mm**2
        if len(spacing) == 2 and img_dim == 3:
            structure = structure[np.newaxis]

        # binary operation
        operation = np.random.choice(self.morph_operations)

        for class_idx in classes:
            # change only this one class
            # we have the -1 since the background is not in the one hot vector
            class_aug = operation(img_one_hot[class_idx - 1], structure)
            img_one_hot[class_idx - 1] = class_aug
            # and for all other classes we remove the fg in this region
            # in case we get intersections from
            for other_class_idx in classes:
                if other_class_idx != class_idx:
                    img_one_hot[other_class_idx - 1][class_aug] = False

        # now we add the background channel again
        img_one_hot = np.concatenate([np.zeros((1, *img.shape)), img_one_hot])

        # from one hot back to interget encoding.
        return np.argmax(img_one_hot, 0)

    def _removal_augmentation(self, img):

        img_dim = len(img.shape)
        mask = np.ones_like(img)

        components = label(img > 0)
        n_components = components.max()

        if self.vol_threshold_removal is not None and self.spacing is not None:
            vol_threshold = self.vol_threshold_removal \
                / np.prod(self.spacing[:img_dim])
        else:
            vol_threshold = self.vol_percentage_removal * np.prod(img.shape)

        for c in range(1, n_components + 1):
            comp = components == c
            if np.sum(comp) < vol_threshold:
                mask[comp] = 0

        return img * mask

    def augment_image(self, img):
        '''
        augment_img(img)
        (nx, ny)
        '''
        global TORCH_WARNING_PRINTED

        is_np, _ = check_type(img)
        if not is_np:
            img = img.cpu().numpy()
            if not TORCH_WARNING_PRINTED:
                print('Warning: Maks augmentations can only be done in '
                      'numpy. Still got a torch tensor as input. Transferring '
                      ' it to the CPU, this kills gradients and might be '
                      'slow.\n')

        if img.max() == 0:
            # no foreground nothing to do!
            return img
        # first collect what we want to do
        self.do_morph = np.random.rand() < self.p_morph
        self.do_removal = np.random.rand() < self.p_removal

        # Let's-a go!
        if self.do_morph:
            img = self._morphological_augmentation(img)
        if self.do_removal:
            img = self._removal_augmentation(img)

        if not is_np:
            img = torch.from_numpy(img).cuda()

        return img

    def augment_sample(self, sample):
        '''
        augment_sample(sample)
        augments only the first image of the sample as we assume single channel
        images like CT
        '''
        for c in self.aug_channels:
            sample[c] = self.augment_image(sample[c])
        return sample

    def augment_batch(self, batch):
        '''
        augment_batch(batch)
        augments every sample of the batch, in each sample only the image in
        the first channel will be augmented as we assume single channel images
        like CT
        '''
        return stack([self.augment_sample(batch[i])
                      for i in range(len(batch))])

    def __call__(self, batch):
        return self.augment_batch(batch)

    def augment_volume(self, volume, is_inverse: bool = False):
        if not is_inverse:
            if len(volume.shape) == 3:
                volume = self.augment_image(volume)
            else:
                volume = self.augment_sample(volume)
        return volume

    def update_prg_trn(self, param_dict, h, indx=None):

        for attr in ['p_morph', 'p_removal', 'radius_mm', 'vol_percentage_removal']:
            if attr in param_dict:
                self.__setattr__(attr, (1 - h) * param_dict[attr][0] + h * param_dict[attr][1])
