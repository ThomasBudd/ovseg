import numpy as np
import torch
import os


class Reconstruction2dDataset(object):

    def __init__(self, vol_ds, batch_size=12, epoch_len=250, image_key='image',
                 projection_key='projection'):
        self.vol_ds = vol_ds
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.image_key = image_key
        self.projection_key = projection_key

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index):
        projs = []
        ims = []
        for _ in range(self.batch_size):
            ind = np.random.randint(len(self.vol_ds))
            path_dict = self.vol_ds.path_dicts[ind]
            proj = np.load(path_dict[self.projection_key], 'r')
            im = np.load(path_dict[self.image_key], 'r')
            z = np.random.randint(im.shape[-1])
            projs.append(proj[np.newaxis, ..., z])
            ims.append(im[np.newaxis, ..., z])
        return np.stack(projs), np.stack(ims)


def ReconstructionDataloader(vol_ds, batch_size, num_workers=None,
                             pin_memory=True, epoch_len=250, image_key='image',
                             projection_key='projection'):
    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else 8
    dataset = Reconstruction2dDataset(vol_ds, batch_size, epoch_len=epoch_len,
                                      image_key=image_key,
                                      projection_key=projection_key)
    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else 8
    worker_init_fn = lambda _: np.random.seed()
    return torch.utils.data.DataLoader(dataset, pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       worker_init_fn=worker_init_fn)
