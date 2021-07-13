import torch
import numpy as np
from ovseg.data.utils import crop_and_pad_image
import os
from time import sleep
from ovseg.data.SegmentationDataloader import SegmentationBatchDataset
from scipy.ndimage.morphology import binary_dilation
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x


class RegionfindingBatchDataset(SegmentationBatchDataset):

    def __init__(self, mask_dist, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # all voxel with a distance smaller than this value are excluded in the loss function
        self.mask_dist = mask_dist
        if len(self.mask_dist) == 2:
            raise NotImplementedError('The region finding dataloader was not '
                                      'implemented yet for 2d operations')
        elif not len(self.mask_dist) == 3:
            raise ValueError('mask_dist must be a tuple of len 3 to indicate the number of '
                             'pixel in each direction that is excluded from the loss.')

        axes = [np.linspace(-1, 1, 2*m+1) for m in self.mask_dist]
        
        # define the ball with radius as in mask_dist
        self.selem = np.sum(np.stack(np.meshgrid(*axes, indexing='ij'))**2, 0) <= 1

    def __getitem__(self, index):

        volume = super().__getitem__(index)
        dtype = volume.dtype
        # get the binary foreground array
        bin_lb = volume[-1] > 0
        # compute the dialated edge
        lb_dial_edge = binary_dilation(bin_lb, self.selem).astype(dtype) - bin_lb.astype(dtype)
        # now keep everything in the loss function, but the dialated edge
        mask = 1 - lb_dial_edge
        mask = mask[np.newaxis]
        if self.return_masks:
            volume = np.concatenate([volume[:-2], mask*volume[-2:-1], volume[-1:]])
        else:
            volume = np.concatenate([volume[:-1], mask, volume[-1:]])
        return volume

def RegionfindingDataloader(vol_ds, patch_size, batch_size, num_workers=None,
                            pin_memory=True, epoch_len=250, p_bias_sampling=0,
                            min_biased_samples=1, augmentation=None, padded_patch_size=None,
                            store_coords_in_ram=True, memmap='r', n_im_channels: int = 1,
                            image_key='image',
                            label_key='label', pred_fps_key=None, n_pred_classes=None,
                            store_data_in_ram=False,
                            return_fp16=True,
                            n_max_volumes=None,
                            bias='fg',
                            mask_key=None,
                            mask_dist=None):
    dataset = RegionfindingBatchDataset(vol_ds=vol_ds, patch_size=patch_size, batch_size=batch_size,
                                        epoch_len=epoch_len, p_bias_sampling=p_bias_sampling,
                                        min_biased_samples=min_biased_samples,
                                        augmentation=augmentation,
                                        padded_patch_size=padded_patch_size,
                                        n_im_channels=n_im_channels,
                                        store_coords_in_ram=store_coords_in_ram,
                                        memmap=memmap, image_key=image_key,
                                        label_key=label_key, pred_fps_key=pred_fps_key,
                                        n_pred_classes=n_pred_classes,
                                        store_data_in_ram=store_data_in_ram,
                                        return_fp16=return_fp16, n_max_volumes=n_max_volumes,
                                        bias=bias, mask_key=mask_key,
                                        mask_dist=mask_dist)
    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else 8
    worker_init_fn = lambda _: np.random.seed()
    sampler = torch.utils.data.SequentialSampler(range(batch_size * epoch_len))
    return torch.utils.data.DataLoader(dataset,
                                       sampler=sampler,
                                       batch_size=batch_size,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       worker_init_fn=worker_init_fn)
