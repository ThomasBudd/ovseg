import torch
import numpy as np
from ovseg.data.utils import crop_and_pad_image
import os
from tqdm import tqdm


class SegmentationBatchDataset(object):

    def __init__(self, vol_ds, patch_size, batch_size, epoch_len=250, p_bias_sampling=0,
                 min_biased_samples=1, augmentation=None, padded_patch_size=None,
                 store_coords_in_ram=True, memmap='r', image_key='image',
                 label_key='label', store_data_in_ram=False, return_fp16=True, n_max_volumes=None):
        self.vol_ds = vol_ds
        self.patch_size = np.array(patch_size)
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.p_bias_sampling = p_bias_sampling
        self.min_biased_samples = min_biased_samples
        self.augmentation = augmentation
        self.store_coords_in_ram = store_coords_in_ram
        self.memmap = memmap
        self.image_key = image_key
        self.label_key = label_key
        self.store_data_in_ram = store_data_in_ram
        self.return_fp16 = return_fp16
        self.n_max_volumes = len(self.vol_ds) if n_max_volumes is None else n_max_volumes

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

        if self.store_data_in_ram:
            self.data = []
            for ind in range(self.n_max_volumes):
                path_dict = self.vol_ds.path_dicts[ind]
                seg = np.load(path_dict[self.label_key])
                im = np.load(path_dict[self.image_key])
                if self.return_fp16:
                    im = im.astype(np.float16)
                self.data.append((im, seg))

        # store coords in ram
        if self.store_coords_in_ram:
            print('Precomputing foreground coordinates to store them in RAM')
            self.coords_list = []
            for ind in tqdm(range(self.n_max_volumes)):
                if self.store_data_in_ram:
                    _, seg = self.data[ind]
                else:
                    seg = np.load(self.vol_ds.path_dicts[ind][self.label_key])
                coords = np.stack(np.where(seg > 0))
                self.coords_list.append(coords)
            print('Done')

    def _get_volume_tuple(self, ind=None):

        if ind is None:
            ind = np.random.randint(self.n_max_volumes)
        if self.store_data_in_ram:
            im, seg = self.data[ind]
        else:
            path_dict = self.vol_ds.path_dicts[ind]
            im = np.load(path_dict[self.image_key], 'r')
            seg = np.load(path_dict[self.label_key], 'r')
        return im, seg

    def __len__(self):
        return self.epoch_len * self.batch_size

    def __getitem__(self, index):

        idx = index % self.batch_size

        if idx < self.min_biased_samples:
            biased_sampling = True
        else:
            biased_sampling = np.random.rand() < self.p_bias_sampling

        ind = np.random.randint(self.n_max_volumes)
        im, seg = self._get_volume_tuple(ind)
        shape = np.array(seg.shape)

        if biased_sampling:
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

        if self.twoD_patches:
            im, seg = im[..., 0], seg[..., 0]

        if self.augmentation is not None:
            sample = np.stack([im, seg])
            sample = self.augmentation.augment_sample(sample)
            im, seg = sample

        if self.return_fp16:
            im, seg = im.astype(np.float16), seg.astype(np.float16)

        return {self.image_key: im[np.newaxis],
                self.label_key: seg[np.newaxis]}


def SegmentationDataloader(vol_ds, patch_size, batch_size, num_workers=None,
                           pin_memory=True, epoch_len=250, p_bias_sampling=1/3,
                           min_biased_samples=1, augmentation=None, padded_patch_size=None,
                           store_coords_in_ram=True, memmap='r',
                           store_data_in_ram=False,
                           return_fp16=True,
                           n_max_volumes=None):
    dataset = SegmentationBatchDataset(vol_ds,
                                       patch_size,
                                       batch_size,
                                       epoch_len=epoch_len,
                                       p_bias_sampling=p_bias_sampling,
                                       min_biased_samples=min_biased_samples,
                                       augmentation=augmentation,
                                       padded_patch_size=padded_patch_size,
                                       store_coords_in_ram=store_coords_in_ram,
                                       store_data_in_ram=store_data_in_ram,
                                       return_fp16=return_fp16,
                                       n_max_volumes=n_max_volumes)
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
