import torch
import numpy as np
from ovseg.data.utils import crop_and_pad_image
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
import os
from time import sleep
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x


class RegionexpertBatchDataset(object):

    def __init__(self, vol_ds, patch_size, batch_size, epoch_len=250, p_bias_sampling=0,
                 min_biased_samples=1, augmentation=None, padded_patch_size=None,
                 n_im_channels: int = 1, store_coords_in_ram=True, memmap='r', image_key='image',
                 label_key='label', store_data_in_ram=False, return_fp16=True, n_max_volumes=None,
                 bias='fg', region_key='region', n_fg_classes=None, *args, **kwargs):
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
        self.region_key = region_key
        self.store_data_in_ram = store_data_in_ram
        self.n_im_channels = n_im_channels
        self.return_fp16 = return_fp16
        self.bias = bias
        
        if self.bias == 'cl_fg':
            assert isinstance(n_fg_classes, int), 'number of classes must be given when bias is cl_fg'
            self.n_fg_classes = n_fg_classes
        else:
            self.n_fg_classes = 1

        # this is only for the progressive training update in SegmentationTraining
        self.mask_key = region_key
        
        if not self.store_coords_in_ram:
            raise NotImplementedError('Please store the coords in RAM.'
                                      'Other option not implemented.')

        self.dtype = np.float16 if self.return_fp16 else np.float32
        if n_max_volumes is None:
            self.n_volumes = len(self.vol_ds)
        else:
            self.n_volumes = np.min([n_max_volumes, len(self.vol_ds)])

        if len(self.patch_size) == 2:
            self.twoD_patches = True
            self.patch_size = np.concatenate([[1], self.patch_size])
        else:
            self.twoD_patches = False

        # overwrite default in case we're not using padding here
        if padded_patch_size is None:
            self.padded_patch_size = self.patch_size
        else:
            self.padded_patch_size = np.array(padded_patch_size)

        self._maybe_store_data_in_ram()

        if len(args) > 0:
            print('Warning, got unused args: {}'.format(args))
        if len(kwargs) > 0:
            print('Warning, got unused kwargs: {}'.format(kwargs))

    def _get_bias_coords(self, volume):

        if self.bias == 'fg':
            return [np.stack(np.where(volume[-1] > 0)).astype(np.int16)]
        elif self.bias == 'cl_fg':
            return [np.stack(np.where(volume[-1] == cl)).astype(np.int16)
                    for cl in range(1, self.n_fg_classes + 1)]
        elif self.bias == 'mask':
            return [np.stack(np.where(volume[-2] > 0)).astype(np.int16)]

    def _maybe_store_data_in_ram(self):
        # maybe cleaning first, just to be sure
        self._maybe_clean_stored_data()

        if self.store_data_in_ram:
            print('Store data in RAM.\n')
            self.data = []
            sleep(1)
            for ind in tqdm(range(self.n_volumes)):
                path_dict = self.vol_ds.path_dicts[ind]
                
                labels = np.load(path_dict[self.label_key]).astype(np.uint8)                
                labels = maybe_add_channel_dim(labels)
                
                im = np.load(path_dict[self.image_key]).astype(self.dtype)
                im = maybe_add_channel_dim(im)

                reg = np.load(path_dict[self.region_key]).astype(self.dtype)
                reg = maybe_add_channel_dim(reg)
                
                self.data.append((im, reg, labels))
                    
        # store coords in ram
        if self.store_coords_in_ram:
            print('Precomputing bias coordinates to store them in RAM.\n')
            self.bias_coords_list = []
            self.contains_fg_list = [[] for _ in range(self.n_fg_classes)]
            self.reg_coords_list = []
            sleep(1)
            for ind in tqdm(range(self.n_volumes)):
                # get label
                if self.store_data_in_ram:
                    labels = self.data[ind][2]
                else:
                    labels = np.load(self.vol_ds.path_dicts[ind][self.label_key])
                labels = maybe_add_channel_dim(labels)

                # get region
                if self.store_data_in_ram:
                    reg = self.data[ind][1]
                else:
                    reg = np.load(self.vol_ds.path_dicts[ind][self.region_key])
                reg = maybe_add_channel_dim(reg)
                
                # in contrast to the segmentation dataloader we're only considering
                # labels inside a region for bias coordinates
                bin_reg = (reg > 0).astype(float)
                lb = labels * bin_reg
                bias_coords = self._get_bias_coords(lb)
                self.bias_coords_list.append(bias_coords)
                
                # save which index has which fg class
                for i in range(self.n_fg_classes):
                    if bias_coords[i].shape[1] > 0:
                        self.contains_fg_list[i].append(ind)
                
                reg_coords = np.stack(np.where(bin_reg[0] > 0)).astype(np.int16)
                self.reg_coords_list.append(reg_coords)
            self.contains_reg_list = [ind for ind, coords in enumerate(self.reg_coords_list)
                                      if coords.shape[1] > 0]
            print('Done')
        else:
            # if we don't store them in ram we will compute them and store them as .npy files
            # in the preprocessed path
            self.contains_bias_list = []
            self.contains_reg_list = []
            self.bias_coords_fol = os.path.join(self.vol_ds.preprocessed_path,
                                                'bias_coordinates_'+self.bias)
            self.reg_coords_fol = os.path.join(self.vol_ds.preprocessed_path,
                                               'reg_coordinates')
            if not os.path.exists(self.bias_coords_fol):
                os.mkdir(self.bias_coords_fol)
            if not os.path.exists(self.reg_coords_fol):
                os.mkdir(self.reg_coords_fol)

            # now we check if come cases are missing in the folder
            print('Checking if all coordinates are stored in '+self.bias_coords_fol+
                  'and '+self.reg_coords_fol)
            for ind, d in enumerate(self.vol_ds.path_dicts):
                case = os.path.basename(d[self.label_key])
                if case not in os.listdir(self.bias_coords_fol):

                    # load label and region from disk
                    labels = maybe_add_channel_dim(np.load(d[self.label_key]))
                    reg = maybe_add_channel_dim(np.load(d[self.region_key]))
                    
                    # in contrast to the segmentation dataloader we're only considering
                    # labels inside a region for bias coordinates
                    bin_reg = (reg > 0).astype(float)
                    lb = labels * bin_reg
                    bias_coords = self._get_bias_coords(lb)
                    np.save(os.path.join(self.bias_coords_fol, case), bias_coords)
                else:
                    bias_coords = np.load(os.path.join(self.bias_coords_fol, case))
                
                # here the coords should be available
                if bias_coords.shape[1] > 0:
                    self.contains_bias_list.append(ind)
                
                if case not in os.listdir(self.reg_coords_fol):

                    reg = np.load(d[self.region_key])

                    # maybe add fourth axis
                    if len(reg.shape) == 4:
                        reg = reg[0]
                    elif not len(reg.shape) == 3:
                        raise ValueError('Got region that is neither 3d nor 4d.')
                    
                    # let's store the coordinates where the previous stage predictes foreground
                    reg_coords = np.stack(np.where(reg > 0)).astype(np.int16)
                    np.save(os.path.join(self.reg_coords_fol, case), reg_coords)
                else:
                    reg_coords = np.load(os.path.join(self.reg_coords_fol, case))
                
                if reg_coords.shape[1] > 0:
                    self.contains_reg_list.append(ind)

        # print how many scans we have with which class
        for c in range(self.n_fg_classes):
            print('Found {} scans with fg {}'.format(len(self.contains_fg_list[c]), c))

        # available classes start from 0
        self.availble_classes = [i for i, l in enumerate(self.contains_fg_list) if len(l) > 0]
        
        # assert len(self.availble_classes) > 0, 'no fg classes found!'
        if len(self.availble_classes) < self.n_fg_classes:
            missing_classes = [i+1 for i, l in enumerate(self.contains_fg_list) if len(l) == 0]
            print('Warning! Some fg classes were not found in this dataset. '
                  'Missing classes: {}'.format(missing_classes))

        sleep(1)

    def _maybe_clean_stored_data(self):
        # delte stuff we stored in RAM
        # first for the full volumes
        if hasattr(self, 'data'):
            for tpl in self.data:
                for arr in tpl:
                    del arr
                del tpl
            del self.data

        # now for the bias coordinates
        if hasattr(self, 'bias_coords_list'):
            for coord in self.bias_coords_list:
                for crds in coord:
                    del crds
                del coord
            del self.bias_coords_list

        if hasattr(self, 'reg_coords_list'):
            for coord in self.reg_coords_list:
                del coord
            del self.reg_coords_list

    def change_folders_and_keys(self, new_folders, new_keys):
        # for progressive training, we might change the folder of image and label data during
        # training if we've stored the rescaled volumes on the hard drive.
        print('Dataloader: chaning keys and folders')
        print('new keys: ', *new_keys)
        print('new folders: ', *new_folders)
        print()
        self.vol_ds.change_folders_and_keys(new_folders, new_keys)
        self._maybe_store_data_in_ram()

    def _get_volume_tuple(self, ind=None):

        if ind is None:
            ind = np.random.randint(self.n_volumes)

        load_from_ram = hasattr(self, 'data')
        if load_from_ram:
            load_from_ram = ind < len(self.data)

        if load_from_ram:
            volumes = self.data[ind]
        else:
            path_dict = self.vol_ds.path_dicts[ind]
            im = np.load(path_dict[self.image_key], 'r')
            reg = np.load(path_dict[self.region_key], 'r')
            labels = np.load(path_dict[self.label_key], 'r')
            volumes = [im, reg, labels]

        # maybe add an additional axis
        volumes = [maybe_add_channel_dim(vol) for vol in volumes]
        
        return volumes

    def _get_random_volume_ind(self, biased_sampling):
        if biased_sampling:
            # when we do biased sampling we have to make sure that the
            # volume we're sampling actually has fg
            if len(self.availble_classes) > 0:
                cl = np.random.choice(self.availble_classes)
                return np.random.choice(self.contains_fg_list[cl]), cl
            else:
                return np.random.choice(self.contains_reg_list), -1
        else:
            # in theory we should only store only preprocessed volumes with regions,
            # we still keep this just to be sure
            return np.random.choice(self.contains_reg_list), -1

    def __len__(self):
        return self.epoch_len * self.batch_size

    def __getitem__(self, index):

        if index % self.batch_size < self.min_biased_samples:
            biased_sampling = True
        else:
            biased_sampling = np.random.rand() < self.p_bias_sampling

        ind, cl = self._get_random_volume_ind(biased_sampling)
        volumes = self._get_volume_tuple(ind)
        shape = np.array(volumes[0].shape)[1:]

        if biased_sampling and cl >= 0:
            # let's get the list of bias coordinates
            if self.store_coords_in_ram:
                # loading from RAM
                coords = self.bias_coords_list[ind][cl]
            else:
                # or hard drive
                case = os.path.basename(self.vol_ds.path_dicts[ind][self.label_key])
                coords = np.load(os.path.join(self.bias_coords_fol, case))[cl]

            # pick a random item from the list and compute the upper left corner of the patch
            n_coords = coords.shape[1]
            coord = coords[:, np.random.randint(n_coords)] - self.patch_size//2
        else:
            # random coordinate uniform from a region
            if self.store_coords_in_ram:
                # loading from RAM
                coords = self.reg_coords_list[ind]
            else:
                # or hard drive
                case = os.path.basename(self.vol_ds.path_dicts[ind][self.label_key])
                coords = np.load(os.path.join(self.reg_coords_fol, case))

            # pick a random item from the list and compute the upper left corner of the patch
            n_coords = coords.shape[1]
            coord = coords[:, np.random.randint(n_coords)] - self.patch_size//2
            
        # round to ensure that the crop does not contain voxel outside of the volume
        coord = np.minimum(np.maximum(coord, 0), shape - self.patch_size)
        # now get the cropped and padded sample

        volume = np.concatenate([crop_and_pad_image(vol,
                                                    coord,
                                                    self.patch_size,
                                                    self.padded_patch_size) for vol in volumes])

        if self.twoD_patches:
            # remove z axis
            volume = volume[:, 0]

        if self.augmentation is not None:
            # in augmentation we need batch style arrays with an additional
            # axes for the batch size
            # the label augmentation expects integer valued predictions as input
            volume = self.augmentation(volume[np.newaxis])[0]

        # cast to fp32 or fp 16
        volume = volume.astype(self.dtype)
        # regions and labels will be passed binary
        volume[-2:] = (volume[-2:] > 0).astype(self.dtype)

        return volume



def RegionexpertDataloader(vol_ds, patch_size, batch_size, num_workers=None,
                           pin_memory=True, epoch_len=250, *args, **kwargs):
    dataset = RegionexpertBatchDataset(vol_ds=vol_ds, patch_size=patch_size, batch_size=batch_size,
                                       epoch_len=epoch_len, *args, **kwargs)
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
