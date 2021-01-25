import numpy as np
import torch
import os


class Reconstruction2dDataset(object):

    def __init__(self,
                 vol_ds,
                 batch_size=12,
                 epoch_len=250,
                 image_key='image',
                 projection_key='projection',
                 store_data_in_ram=False,
                 return_fp16=True,
                 n_max_volumes=None):
        self.vol_ds = vol_ds
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.image_key = image_key
        self.projection_key = projection_key
        self.store_data_in_ram = store_data_in_ram
        self.return_fp16 = return_fp16
        self.n_max_volumes = len(self.vol_ds) if n_max_volumes is None else n_max_volumes

        if self.store_data_in_ram:
            self.data = []
            for ind in range(self.n_max_volumes):
                path_dict = self.vol_ds.path_dicts[ind]
                proj = np.load(path_dict[self.projection_key])
                im = np.load(path_dict[self.image_key])
                if self.return_fp16:
                    proj, im = proj.astype(np.float16), im.astype(np.float16)
                self.data.append((proj, im))

    def _get_volume_tuple(self, ind=None):

        if ind is None:
            ind = np.random.randint(self.n_max_volumes)
        if self.store_data_in_ram:
            proj, im = self.data[ind]
        else:
            path_dict = self.vol_ds.path_dicts[ind]
            proj = np.load(path_dict[self.projection_key], 'r')
            im = np.load(path_dict[self.image_key], 'r')
        return proj, im

    def __len__(self):
        return self.epoch_len * self.batch_size

    def __getitem__(self, index):
        proj, im = self._get_volume_tuple()
        z = np.random.randint(im.shape[-1])
        proj = proj[np.newaxis, ..., z]
        im = im[np.newaxis, ..., z]
        if self.return_fp16:
            proj, im = proj.astype(np.float16), im.astype(np.float16)
        return {self.image_key: im, self.projection_key: proj}


def ReconstructionDataloader(vol_ds, batch_size, num_workers=None,
                             pin_memory=True, epoch_len=250, image_key='image',
                             projection_key='projection',
                             store_data_in_ram=False,
                             return_fp16=True,
                             n_max_volumes=None):
    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else 8
    dataset = Reconstruction2dDataset(vol_ds, batch_size, epoch_len=epoch_len,
                                      image_key=image_key,
                                      projection_key=projection_key,
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
