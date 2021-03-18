import numpy as np
import torch
import os
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x


class Reconstruction2dDataset(object):

    def __init__(self,
                 vol_ds,
                 batch_size=12,
                 epoch_len=250,
                 image_key='image',
                 projection_key='projection',
                 store_data_in_ram=False,
                 return_fp16=True,
                 n_max_volumes=None,
                 n_bias=0):
        self.vol_ds = vol_ds
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.image_key = image_key
        self.projection_key = projection_key
        self.store_data_in_ram = store_data_in_ram
        self.return_fp16 = return_fp16
        self.n_max_volumes = len(self.vol_ds) if n_max_volumes is None else n_max_volumes
        self.n_bias = n_bias

        if self.store_data_in_ram:
            print('Putting data in RAM.\n')
            self.data = []
            for ind in tqdm(range(self.n_max_volumes)):
                path_dict = self.vol_ds.path_dicts[ind]
                proj = np.load(path_dict[self.projection_key])
                im = np.load(path_dict[self.image_key])
                if self.return_fp16:
                    proj, im = proj.astype(np.float16), im.astype(np.float16)
                self.data.append((proj, im))

        if self.n_bias > 0:
            self.bias_slices = []
            lbp = os.path.join(self.vol_ds.preprocessed_path, 'labels')
            for pd in self.vol_ds.path_dicts:
                case = os.path.basename(pd[self.projection_key])
                lb = np.load(os.path.join(lbp, case))
                lb = (np.sum(lb, (0, 1)) > 0).astype(np.int16)
                self.bias_slices.append(lb)

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
        ind = np.random.randint(self.n_max_volumes)
        proj, im = self._get_volume_tuple(ind)
        if index % self.batch_size < self.n_bias:
            bias = self.bias_slices[ind]
            z = np.random.choice(bias)
        else:
            z = np.random.randint(im.shape[-1])
        proj = proj[np.newaxis, ..., z]
        im = im[np.newaxis, ..., z]
        if self.return_fp16:
            proj, im = proj.astype(np.float16), im.astype(np.float16)
        else:
            proj, im = proj.astype(np.float32), im.astype(np.float32)
        return {self.image_key: im, self.projection_key: proj}


def ReconstructionDataloader(vol_ds, batch_size, num_workers=None,
                             pin_memory=True, epoch_len=250, image_key='image',
                             projection_key='projection',
                             store_data_in_ram=False,
                             return_fp16=True,
                             n_max_volumes=None,
                             n_bias=0):
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
