import torch
import numpy as np
import os


class JoinedBatchDataset(object):

    def __init__(self, vol_ds, batch_size, patch_size, epoch_len=250, p_fg=0,
                 mn_fg=3, store_coords_in_ram=True, memmap='r',
                 projection_key='projection', image_key='image',
                 label_key='label', spacing_key='spacing'):
        self.vol_ds = vol_ds
        self.batch_size = batch_size
        self.patch_size = np.array(patch_size)
        self.epoch_len = epoch_len
        self.p_fg = p_fg
        self.mn_fg = mn_fg
        self.store_coords_in_ram = store_coords_in_ram
        self.memmap = memmap
        self.image_key = image_key
        self.label_key = label_key
        self.projection_key = projection_key
        self.spacing_key = spacing_key

        # store coords in ram
        if self.store_coords_in_ram:
            print('Precomputing foreground coordinates to store them in RAM')
            self.coords_list = []
            for ind in range(len(self.vol_ds)):
                data_dict = self.vol_ds[ind]
                seg = data_dict['label']
                if seg.max() > 0:
                    coords = np.stack(np.where(np.sum(seg, (0, 1)) > 0)[0])
                else:
                    coords = np.array([])
                self.coords_list.append(coords)
            print('Done')

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index=None):
        # makes a new batch and stores it
        # we're doing this so that the __getitem__ function can return samples
        # instead of batches
        projs = []
        ims = []
        segs = []
        xycoords = []
        spacings = []
        # draw the number of fg patches in the batch
        n_fg_samples = self.mn_fg + np.sum([np.random.rand() < self.p_fg
                                            for _ in range(self.batch_size -
                                                           self.mn_fg)])

        for b in range(self.batch_size):

            # draw random index
            ind = np.random.randint(len(self.vol_ds))
            # get the dict with the pathes to the desired data
            path_dict = self.vol_ds.path_dicts[ind]

            # load the memory maps of the data
            proj = np.load(path_dict[self.projection_key], self.memmap)
            im = np.load(path_dict[self.image_key], self.memmap)
            seg = np.load(path_dict[self.label_key], self.memmap)
            spacing = np.load(path_dict[self.spacing_key])

            # how many fg samples do we alreay have in the batch?
            k_fg_samples = np.sum([np.max(samp > 0) for samp in segs])
            if k_fg_samples < n_fg_samples:
                # if we're not there let's choose a center coordinate
                # that contains fg
                if self.store_coords_in_ram:
                    coords = self.coords_list[ind]
                else:
                    # or not!
                    if seg.max() > 0:
                        coords = np.stack(np.where(np.sum(seg, (0, 1)) > 0)[0])
                    else:
                        coords = np.array([])
                n_coords = len(coords)
                if n_coords > 0:
                    zcoord = coords[np.random.randint(n_coords)]
                else:
                    # random coordinate
                    zcoord = np.random.randint(seg.shape[-1])
            else:
                # random coordinate
                zcoord = np.random.randint(seg.shape[-1])
            # now get the cropped and padded sample
            projs.append(proj[np.newaxis, ..., zcoord])
            ims.append(im[np.newaxis, ..., zcoord])
            # since the images have different shapes we first have to
            # pad or crop them and keep the crop coordinate
            sample = seg[..., zcoord]
            seg_sl, xycoord = self._crop_or_pad(sample)
            segs.append(seg_sl[np.newaxis])
            xycoords.append(xycoord)
            spacings.append(spacing)

        # stack up in first dim except for the segmentations as they have
        # different resolutions
        batch = {'projection': np.stack(projs), 'image': np.stack(ims),
                 'label': np.stack(segs), 'xycoords': np.stack(xycoords),
                 'spacing': np.stack(spacings)}

        return batch

    def _crop_or_pad(self, sample):
        shape = np.array(sample.shape)
        if np.all(shape <= self.patch_size):
            # we're smaller let's pad the arrays
            pad = [[0, p - s] for p, s in zip(self.patch_size, shape)]
            sample = np.pad(sample, pad)
            xycoord = np.array([0, 0])
        elif np.all(shape > self.patch_size):
            # the sample is larger, let's do a random crop!
            xycoord = np.random.randint(shape - self.patch_size + 1)
            sample = sample[xycoord[0]:xycoord[0]+self.patch_size[0],
                            xycoord[1]:xycoord[1]+self.patch_size[1]]
        else:
            raise ValueError('Something weird happend when try to crop or '
                             'pad! Got shape {} and patch size '
                             '{}'.format(shape, self.patch_size))
        return sample, xycoord


def JoinedDataloader(vol_ds, batch_size, patch_size, num_workers=None,
                     pin_memory=True, epoch_len=250, p_fg=1/3,
                     mn_fg=1, store_coords_in_ram=True, memmap='r'):
    dataset = JoinedBatchDataset(vol_ds, batch_size, patch_size,
                                 epoch_len=epoch_len,
                                 p_fg=p_fg, mn_fg=mn_fg,
                                 store_coords_in_ram=store_coords_in_ram)
    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else 8
    worker_init_fn = lambda _: np.random.seed()
    return torch.utils.data.DataLoader(dataset, pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       worker_init_fn=worker_init_fn)
