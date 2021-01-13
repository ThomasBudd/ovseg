import torch
import numpy as np
from ovseg.data.utils import crop_and_pad_image
import os
from tqdm import tqdm


class SegmentationBatchDataset(object):

    def __init__(self, vol_ds, patch_size, batch_size, epoch_len=250, p_fg=0,
                 mn_fg=1, augmentation=None, padded_patch_size=None,
                 store_coords_in_ram=True, memmap='r', image_key='image',
                 label_key='label'):
        self.vol_ds = vol_ds
        self.patch_size = np.array(patch_size)
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.p_fg = p_fg
        self.mn_fg = mn_fg
        self.augmentation = augmentation
        self.store_coords_in_ram = store_coords_in_ram
        self.memmap = memmap
        self.image_key = image_key
        self.label_key = label_key

        if len(self.patch_size) == 2:
            self.twoD_patches = True
            self.patch_size = np.concatenate([self.patch_size, [1]])
        else:
            self.twoD_patches = False

        # overwrite default in case we're not using padding here
        if padded_patch_size is None:
            self.padded_patch_size = self.patch_size
        else:
            self.padded_patch_size = np.array(self.padded_patch_size)

        # store coords in ram
        if self.store_coords_in_ram:
            print('Precomputing foreground coordinates to store them in RAM')
            self.coords_list = []
            for ind in tqdm(range(len(self.vol_ds))):
                data_dict = self.vol_ds[ind]
                seg = data_dict[self.label_key]
                coords = np.stack(np.where(seg > 0))
                self.coords_list.append(coords)
            print('Done')

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index=None):
        # makes a new batch and stores it
        # we're doing this so that the __getitem__ function can return samples
        # instead of batches
        batch = []
        # draw the number of fg patches in the batch
        n_fg_samples = self.mn_fg + np.sum([np.random.rand() < self.p_fg
                                            for _ in range(self.batch_size -
                                                           self.mn_fg)])

        for b in range(self.batch_size):
            ind = np.random.randint(len(self.vol_ds))
            path_dict = self.vol_ds.path_dicts[ind]
            im = np.load(path_dict[self.image_key], self.memmap)
            seg = np.load(path_dict[self.label_key], self.memmap)
            shape = np.array(seg.shape)
            # how many fg samples do we alreay have in the batch?
            k_fg_samples = np.sum([np.max(samp[1] > 0) for samp in batch])
            if k_fg_samples < n_fg_samples:
                # if we're not there let's choose a center coordinate
                # that contains fg
                if self.store_coords_in_ram:
                    coords = self.coords_list[ind]
                else:
                    # or not!
                    coords = np.stack(np.where(seg > 0))
                n_coords = coords.shape[1]
                if n_coords > 0:
                    coord = coords[:, np.random.randint(n_coords)] \
                        - self.patch_size//2
                else:
                    # random coordinate
                    coord = np.random.randint(
                        np.maximum(shape - self.patch_size+1, 1))
            else:
                # random coordinate
                coord = np.random.randint(
                    np.maximum(shape - self.patch_size+1, 1))
            coord = np.minimum(np.maximum(coord, 0), shape - self.patch_size)
            # now get the cropped and padded sample
            im = crop_and_pad_image(im, coord, self.patch_size,
                                    self.padded_patch_size)
            seg = crop_and_pad_image(seg, coord, self.patch_size,
                                     self.padded_patch_size)
            batch.append(np.stack([im, seg]))
        # stack up in first dim
        batch = np.stack(batch)
        # for the 2d case have remove the last axes
        if self.twoD_patches:
            batch = batch[..., 0]

        if self.augmentation is not None:
            batch = self.augmentation.augment_batch(batch)

        return batch


def SegmentationDataloader(vol_ds, patch_size, batch_size, num_workers=None,
                           pin_memory=True, epoch_len=250, p_fg=1/3,
                           mn_fg=1, augmentation=None, padded_patch_size=None,
                           store_coords_in_ram=True, memmap='r'):
    dataset = SegmentationBatchDataset(vol_ds, patch_size, batch_size,
                                       epoch_len=epoch_len, p_fg=p_fg,
                                       mn_fg=mn_fg, augmentation=augmentation,
                                       padded_patch_size=padded_patch_size,
                                       store_coords_in_ram=store_coords_in_ram)
    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else 8
    worker_init_fn = lambda _: np.random.seed()
    return torch.utils.data.DataLoader(dataset, pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       worker_init_fn=worker_init_fn)
