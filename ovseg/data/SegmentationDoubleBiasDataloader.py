import torch
import numpy as np
from ovseg.data.utils import crop_and_pad_image
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from ovseg.utils.io import read_nii
import os
import nibabel as nib
from time import sleep
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x


def torch_resize(label, pred):
    
    pred_gpu = torch.from_numpy(pred[np.newaxis,np.newaxis]).cuda()
    
    size = label.shape
    if len(size) == 4:
        size = size[0]
    
    pred_rsz = torch.nn.functional.interpolate(pred_gpu, size)
    
    return pred_rsz[0,0].cpu().numpy()
    

class SegmentationDoubleBiasBatchDataset(object):

    def __init__(self, vol_ds, patch_size, batch_size, epoch_len=250,
                 n_bias1=1,n_bias2=1, prev_preds:list = [],
                 augmentation=None, padded_patch_size=None,
                 n_im_channels: int = 1, memmap='r', image_key='image',
                 label_key='label', store_data_in_ram=False, return_fp16=True, n_max_volumes=None,
                 bias1='fg', n_fg_classes=None, lb_classes=None, *args, **kwargs):
        self.vol_ds = vol_ds
        self.patch_size = np.array(patch_size)
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.n_bias1 = n_bias1
        self.n_bias2 = n_bias2
        self.prev_preds = prev_preds
        self.augmentation = augmentation
        self.memmap = memmap
        self.image_key = image_key
        self.label_key = label_key
        self.store_data_in_ram = store_data_in_ram
        self.n_im_channels = n_im_channels
        self.return_fp16 = return_fp16
        self.bias1 = bias1
        self.n_fg_classes = n_fg_classes
        self.lb_classes = lb_classes
        
        if self.bias1 == 'cl_fg':
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
            
            

        assert len(self.prev_preds) > 0, 'Need infos for previous predictions'
        self.path_to_previous_preds = os.path.join(os.environ['OV_DATA_BASE'],
                                                   'predictions',
                                                   *self.prev_preds)

        self._maybe_store_data_in_ram()

        if len(args) > 0:
            print('Warning, got unused args: {}'.format(args))
        if len(kwargs) > 0:
            print('Warning, got unused kwargs: {}'.format(kwargs))



    def _get_bias_coords(self, labels, pred):

        if self.bias1 == 'fg':
            coords1 = [np.stack(np.where(labels[-1] > 0)).astype(np.int16)]
        elif self.bias1 == 'cl_fg':
            coords1 = [np.stack(np.where(labels[-1] == cl)).astype(np.int16)
                       for cl in range(1, self.n_fg_classes + 1)]
        
        bin_lb = (labels[-1] > 0).astype(float)
        bin_pred = (pred > 0).astype(float)
        coords2 = np.stack(np.where(np.abs(bin_lb-bin_pred) > 0)).astype(np.int16)
        
        return [coords1, coords2]

    def _get_bias2_weight(self, labels, pred):
        
        lb = labels[-1]
        
        w = 0
        for i, cl in enumerate(self.lb_classes):
            bin_lb = (lb == i+1).astype(float)
            bin_pred = (pred ==cl).astype(float)
            w += 1 - 2*(np.sum(bin_lb*bin_pred) + 1) / (np.sum(bin_lb + bin_pred) + 1)
        
        return w
        

    def _get_prev_pred(self, d):
        case = os.path.basename(d[self.label_key]).split('.')[0]
        pred, _, _ = read_nii(os.path.join(self.path_to_previous_preds,
                                           case+'.nii.gz'))
        return pred

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
                
                self.data.append((im, labels))
                    
        # store coords in ram
        print('Precomputing bias coordinates to store them in RAM.\n')
        self.coords_list = []
        self.bias2_weights = []
        self.contains_fg_list = [[] for _ in range(self.n_fg_classes)]
        sleep(1)
        for ind in tqdm(range(self.n_volumes)):
            if self.store_data_in_ram:
                labels = self.data[ind][1]
            else:
                labels = np.load(self.vol_ds.path_dicts[ind][self.label_key])
            # ensure 4d array
            labels = maybe_add_channel_dim(labels)
            
            # get prev prediction in right shape
            pred = self._get_prev_pred(self.vol_ds.path_dicts[ind])
            pred = torch_resize(labels, pred)
            
            coords = self._get_bias_coords(labels, pred)
            self.coords_list.append(coords)

            self.bias2_weights.append(self._get_bias2_weight(labels, pred))

            # save which index has which fg class
            for i in range(self.n_fg_classes):
                if coords[0][i].shape[1] > 0:
                    self.contains_fg_list[i].append(ind)
        print('Done')
        
        # print how many scans we have with which class
        for c in range(self.n_fg_classes):
            print('Found {} scans with fg {}'.format(len(self.contains_fg_list[c]), c))

        # available classes start from 0
        self.availble_classes = [i for i, l in enumerate(self.contains_fg_list) if len(l) > 0]
        
        if len(self.availble_classes) < self.n_fg_classes:
            missing_classes = [i+1 for i, l in enumerate(self.contains_fg_list) if len(l) == 0]
            print('Warning! Some fg classes were not found in this dataset. '
                  'Missing classes: {}'.format(missing_classes))
        
        # now make the bias2_weight a probability distribution
        self.bias2_weights = np.array(self.bias2_weights)
        self.bias2_weights /= np.sum(self.bias2_weights)
        
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
        if hasattr(self, 'coords_list'):
            for coord in self.coords_list:
                for crds in coord:
                    del crds
                del coord
            del self.coords_list

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
            labels = np.load(path_dict[self.label_key], 'r')
            volumes = [im, labels]

        # maybe add an additional axis
        volumes = [maybe_add_channel_dim(vol) for vol in volumes]
        
        return volumes

    def _get_random_volume_ind(self, bias):
        
        if bias == 0:
            return np.random.randint(self.n_volumes), -1
        
        elif bias == 1:
            # when we do biased sampling we have to make sure that the
            # volume we're sampling actually has fg
            if len(self.availble_classes) > 0:
                cl = np.random.choice(self.availble_classes)
                return np.random.choice(self.contains_fg_list[cl]), cl
            else:
                return np.random.randint(self.n_volumes), -1
            
        else:
            return np.random.choice(list(range(self.n_volumes)), p=self.bias2_weights), -1

    def __len__(self):
        return self.epoch_len * self.batch_size

    def __getitem__(self, index):

        
        rel_indx = index % self.batch_size
        if rel_indx < self.n_bias1:
            bias = 1
        elif rel_indx < self.n_bias1 + self.n_bias2:
            bias = 2
        else:
            bias = 0

        ind, cl = self._get_random_volume_ind(bias)
        volumes = self._get_volume_tuple(ind)
        shape = np.array(volumes[0].shape)[1:]

        if bias == 1 and cl >= 0:
            # let's get the list of bias coordinates
            # loading from RAM
            coords = self.coords_list[ind][0][cl]

            # pick a random item from the list and compute the upper left corner of the patch
            n_coords = coords.shape[1]
            coord = coords[:, np.random.randint(n_coords)] - self.patch_size//2
        elif bias == 2 and (self.coords_list[ind][1]).shape[1] > 0:
            # loading from RAM
            coords = self.coords_list[ind][1]

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



def SegmentationDoubleBiasDataloader(vol_ds, patch_size, batch_size, num_workers=None,
                           pin_memory=True, epoch_len=250, *args, **kwargs):
    dataset = SegmentationDoubleBiasBatchDataset(vol_ds=vol_ds,
                                                 patch_size=patch_size,
                                                 batch_size=batch_size,
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
