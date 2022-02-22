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


class SegmentationBatchDatasetV2(object):

    def __init__(self, vol_ds, patch_size, batch_size, epoch_len=250, p_bias_sampling=0,
                 min_biased_samples=1, augmentation=None, padded_patch_size=None,
                 n_im_channels: int = 1, p_weighted_volume_sampling=0,
                 store_data_in_ram=False, return_fp16=True, n_max_volumes=None,
                 bias='fg', n_fg_classes=None, *args, **kwargs):
        self.vol_ds = vol_ds
        self.patch_size = np.array(patch_size)
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.p_bias_sampling = p_bias_sampling
        self.min_biased_samples = min_biased_samples
        self.augmentation = augmentation
        self.store_data_in_ram = store_data_in_ram
        self.n_im_channels = n_im_channels
        self.p_weighted_volume_sampling = p_weighted_volume_sampling
        self.return_fp16 = return_fp16
        self.bias = bias
        self.n_fg_classes = n_fg_classes
        
        if self.bias in ['cl_fg', 'error']:
            assert isinstance(self.n_fg_classes, int)
            assert self.n_fg_classes > 0
        else:
            # does not need to be true, but makes our life easier
            self.n_fg_classes = 1

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

        # now check if we're considerin masks and previous predictions
        self.has_masks = 'mask' in self.vol_ds.keys
        self.has_pp = 'prev_pred' in self.vol_ds.keys

        self._maybe_store_data_in_ram()

        if len(args) > 0:
            print('Warning, got unused args: {}'.format(args))
        if len(kwargs) > 0:
            print('Warning, got unused kwargs: {}'.format(kwargs))


        # now this is needed for the progressive learning
        self.image_key = 'image'
        self.label_key = 'label'
        if self.has_masks:
            self.mask_key = 'mask'
        if self.has_pp:
            self.prev_pred_key = 'prev_pred'

    def _compute_bias_coordinates_and_weight(self, volumes):
        
        if self.bias == 'fg':
            coords = [np.stack(np.where(volumes[-1] > 0)).astype(np.int16)]
            weight = 1
        elif self.bias == 'cl_fg':
            coords = [np.stack(np.where(volumes[-1] == cl)).astype(np.int16)
                    for cl in range(1, self.n_fg_classes + 1)]
            weight = 1
        elif self.bias == 'error':
            
            assert self.has_pp, 'No previous predictions were given, can\'t sample error regions'
            
            coords = []
            weight = 0
            
            lb = volumes[-1]
            pp = volumes[1]
            
            for cl in range(1, self.n_fg_classes):
                
                lb_cl = (lb == cl).astype(float)
                pp_cl = (pp == cl).astype(float)
                
                error = (lb_cl - pp_cl).abs() > 0
                
                coords.append(np.stack(np.where(error)).astype(np.int16))
                if lb_cl.max() > 0 or pp_cl.max() > 0:
                    weight += 1 - 2*np.sum(lb_cl * pp_cl) / np.sum(pp_cl + lb_cl)
                else:
                    weight += 0
            
            # not needed, but no weight is again in [0,1]
            weight /= self.n_fg_classes
            
        return coords, weight

    def _maybe_store_data_in_ram(self):
        # maybe cleaning first, just to be sure
        self._maybe_clean_stored_data()

        if self.store_data_in_ram:
            print('Store data in RAM.\n')
            self.data = []
            sleep(1)
            for ind in tqdm(range(self.n_volumes)):                
                volumes = self._read_volume_tuple(ind, mmap_mode=None)
                self.data.append(volumes)
                    
        # store coords in ram
        print('Precomputing bias coordinates to store them in RAM.\n')
        self.bias_coords_list = []
        self.contains_bias_list = [[] for _ in range(self.n_fg_classes)]
        self.weight_list = []
        
        if self.has_masks:
            # if we have masks we only want to sample inside them
            self.mask_coords_list = []
            self.contains_mask_list = []
        
        sleep(0.1)
        for ind in tqdm(range(self.n_volumes)):
            volumes = self._get_volume_tuple(ind)
            
            coords, weight = self._compute_bias_coordinates_and_weights(volumes)
            self.bias_coords_list.append(coords)
            self.weight_list.append(weight)

            # save which index has which fg class
            for i in range(self.n_fg_classes):
                if coords[i].shape[1] > 0:
                    self.contains_bias_list[i].append(ind)
            
            # now if we have masks we will only sample inside them
            if self.has_masks:
                mask_coords = np.stack(np.where(volumes[-2] > 0)).astype(np.int16)
                self.mask_coords_list.append(mask_coords)
                if mask_coords.shape[1] > 0:
                    self.contains_mask_list.append(ind)
        
        self.weight_list = np.array(self.weight_list) / np.sum(self.weight_list)
        
        print('Done')
        
        # print how many scans we have with which class
        for c in range(self.n_fg_classes):
            print('Found {} scans with fg {}'.format(len(self.contains_bias_list[c]), c))

        # available classes start from 0
        self.availble_classes = [i for i, l in enumerate(self.contains_bias_list) if len(l) > 0]
        
        if len(self.availble_classes) < self.n_fg_classes:
            missing_classes = [i+1 for i, l in enumerate(self.contains_bias_list) if len(l) == 0]
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
        
        
        # now for the bias coordinates
        if hasattr(self, 'mask_coords_list'):
            for coord in self.mask_coords_list:
                for crds in coord:
                    del crds
                del coord
            del self.mask_coords_list

    def change_folders_and_keys(self, new_folders, new_keys):
        # for progressive training, we might change the folder of image and label data during
        # training if we've stored the rescaled volumes on the hard drive.
        print('Dataloader: chaning keys and folders')
        print('new keys: ', *new_keys)
        print('new folders: ', *new_folders)
        print()
        self.vol_ds.change_folders_and_keys(new_folders, new_keys)
        self._maybe_store_data_in_ram()

    def _read_volume_tuple(self, ind, mmap_mode):
        path_dict = self.vol_ds.path_dicts[ind]
        
        # load image
        volumes = [np.load(path_dict['image'], mmap_mode)]
        
        if self.has_pp:
            volumes.append(np.load(path_dict['prev_pred'], mmap_mode))
        
        if self.has_mask:
            volumes.append(np.load(path_dict['mask'], mmap_mode))
            
        volumes.append(np.load(path_dict['label'], mmap_mode))
        
        return volumes

    def _get_volume_tuple(self, ind=None):

        if ind is None:
            ind = np.random.randint(self.n_volumes)

        load_from_ram = hasattr(self, 'data')
        if load_from_ram:
            load_from_ram = ind < len(self.data)

        if load_from_ram:
            volumes = self.data[ind]
        else:
            volumes = self._read_volume_tuple(ind, mmap_mode='r')

        # maybe add an additional axis
        volumes = [maybe_add_channel_dim(vol) for vol in volumes]
        
        return volumes

    def _get_random_volume_ind(self, biased_sampling):
        # returns the volume index and a random class
        
        if biased_sampling:
            
            if self.bias == 'error':
                # first pick a volume according to the weight
                if np.random.rand() < self.p_weighted_volume_sampling:
                    ind = np.random.choice(range(self.n_volumes), 
                                           p=self.weight_list)
                else:
                    ind = np.random.randint(self.n_volumes)
            
                classes_present = [cl for cl in range(self.availble_classes) if ind in self.contains_bias_list[cl]]
                cl = np.random.choice(classes_present)
                return ind, cl
            else:
                # for the other sampling schemes this is enough
                if len(self.availble_classes) > 0:
                    cl = np.random.choice(self.availble_classes)
                    return np.random.choice(self.contains_bias_list[cl]), cl
                else:
                    return np.random.randint(self.n_volumes), -1
        else:
            # when we do no biased sampling we just pick the volume
            # uniform at random
            return np.random.randint(self.n_volumes), -1

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
            # let's get the list of bias coordinates from RAM
            coords = self.bias_coords_list[ind][cl]
            
            # pick a random item from the list and compute the upper left corner of the patch
            n_coords = coords.shape[1]
            coord = coords[:, np.random.randint(n_coords)] - self.patch_size//2
        else:
            
            if self.has_masks:
                # let's get the list of bias coordinates from RAM
                coords = self.mask_coords_list[ind][cl]
                
                # pick a random item from the list and compute the upper left corner of the patch
                n_coords = coords.shape[1]
                coord = coords[:, np.random.randint(n_coords)] - self.patch_size//2
            else:
                # random coordinate uniform from the whole volume
                coord = np.random.randint(np.maximum(shape - self.patch_size+1, 1))
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

        return volume.astype(self.dtype)



def SegmentationDataloaderV2(vol_ds, patch_size, batch_size, num_workers=None,
                           pin_memory=True, epoch_len=250, *args, **kwargs):
    dataset = SegmentationBatchDatasetV2(vol_ds=vol_ds, patch_size=patch_size, batch_size=batch_size,
                                       epoch_len=epoch_len, *args, **kwargs)
    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else 5
    worker_init_fn = lambda _: np.random.seed()
    sampler = torch.utils.data.SequentialSampler(range(batch_size * epoch_len))
    return torch.utils.data.DataLoader(dataset,
                                       sampler=sampler,
                                       batch_size=batch_size,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       worker_init_fn=worker_init_fn)
