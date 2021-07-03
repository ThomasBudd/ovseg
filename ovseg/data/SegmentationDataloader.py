import torch
import numpy as np
from ovseg.data.utils import crop_and_pad_image
import os
from time import sleep
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x


class SegmentationBatchDataset(object):

    def __init__(self, vol_ds, patch_size, batch_size, epoch_len=250, p_bias_sampling=0,
                 min_biased_samples=1, augmentation=None, padded_patch_size=None,
                 n_im_channels: int = 1, store_coords_in_ram=True, memmap='r', image_key='image',
                 label_key='label', pred_fps_key=None, n_pred_classes=None,
                 store_data_in_ram=False, return_fp16=True, n_max_volumes=None, bias='fg'):
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
        self.pred_fps_key = pred_fps_key
        self.n_pred_classes = n_pred_classes
        self.store_data_in_ram = store_data_in_ram
        self.n_im_channels = n_im_channels
        self.return_fp16 = return_fp16
        self.bias = bias
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

        # now some cascade stuff
        self.is_cascade = self.pred_fps_key is not None
        if self.is_cascade:
            assert isinstance(self.n_pred_classes, int), 'n_pred_classes must be an integer'
        self._maybe_store_data_in_ram()

    def _get_bias_coords(self, seg, pred_fps=None):

        if self.bias == 'fg':
            return np.stack(np.where(seg > 0)).astype(np.int16)
        elif self.bias == 'mv':
            mv = 0
            for c in range(self.n_pred_classes):
                seg_c = (seg == c).astype(float)
                pred_c = (pred_fps == c).astype(float)
                mv += seg_c * (1 - pred_c)
            return np.stack(np.where(mv > 0)).astype(np.int16)

    def _maybe_store_data_in_ram(self):
        # maybe cleaning first, just to be sure
        self._maybe_clean_stored_data()

        if self.store_data_in_ram:
            print('Store data in RAM.\n')
            self.data = []
            sleep(1)
            for ind in tqdm(range(self.n_volumes)):
                path_dict = self.vol_ds.path_dicts[ind]
                seg = np.load(path_dict[self.label_key])
                im = np.load(path_dict[self.image_key])
                if self.return_fp16:
                    im = im.astype(np.float16)
                if self.is_cascade:
                    pred_fps = np.load(path_dict[self.pred_fps_key])
                    self.data.append((im, pred_fps, seg))
                else:
                    self.data.append((im, seg))

        # store coords in ram
        if self.store_coords_in_ram:
            print('Precomputing bias coordinates to store them in RAM.\n')
            self.coords_list = []
            sleep(1)
            for ind in tqdm(range(self.n_volumes)):
                if self.store_data_in_ram:
                    seg = self.data[ind][-1]
                    if self.is_cascade:
                        pred_fps = self.data[ind][1]
                    else:
                        pred_fps = None
                else:
                    seg = np.load(self.vol_ds.path_dicts[ind][self.label_key])
                    if self.is_cascade:
                        pred_fps = np.load(self.vol_ds.path_dicts[ind][self.pred_fps_key])
                    else:
                        pred_fps = None
                if len(seg.shape) == 4:
                    seg = seg[0]
                elif not len(seg.shape) == 3:
                    raise ValueError('Got segmentation mask that is neither 3d nor 4d.')
                coords = self._get_bias_coords(seg, pred_fps)
                self.coords_list.append(coords)
            self.contains_fg_list = [ind for ind, coords in enumerate(self.coords_list)
                                    if len(coords) > 0]
            print('Done')
        else:
            # if we don't store them in ram we will compute them and store them as .npy files
            # in the preprocessed path
            self.contains_fg_list = []
            self.bias_coords_fol = os.path.join(self.vol_ds.preprocessed_path,
                                                'bias_coordinates_'+self.bias)
            if not os.path.exists(self.bias_coords_fol):
                os.mkdir(self.bias_coords_fol)

            # now we check if come cases are missing in the folder
            print('Checking if all bias coordinates are stored in '+self.bias_coords_fol)
            for ind, d in enumerate(self.vol_ds.path_dicts):
                case = os.path.basename(d[self.label_key])
                if case not in os.listdir(self.bias_coords_fol):
                    lb = np.load(d[self.label_key])
                    if self.is_cascade:
                        pred_fps = np.load(d[self.pred_fps_key])
                    else:
                        pred_fps = None
                    coords = self._get_bias_coords(lb, pred_fps)
                    np.save(os.path.join(self.bias_coords_fol, case), coords)
                else:
                    coords = np.load(os.path.join(self.bias_coords_fol, case))
                if len(coords) > 0:
                    self.contains_fg_list.append(ind)

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
            if self.is_cascade:
                im, pred_fps, seg = self.data[ind]
            else:
                im, seg = self.data[ind]
        else:
            path_dict = self.vol_ds.path_dicts[ind]
            im = np.load(path_dict[self.image_key], 'r')
            seg = np.load(path_dict[self.label_key], 'r')
            if self.is_cascade:
                pred_fps = np.load(path_dict[self.pred_fps_key], 'r')

        # maybe add an additional axis
        if len(im.shape) == 3:
            im = im[np.newaxis]
        if len(seg.shape) == 3:
            seg = seg[np.newaxis]
        if self.is_cascade:
            if len(pred_fps.shape) == 3:
                pred_fps = pred_fps[np.newaxis]
        
        if self.is_cascade:
            return im, pred_fps, seg
        else:
            return im, seg

    def _get_random_volume_ind(self, biased_sampling):
        if biased_sampling:
            # when we do biased sampling we have to make sure that the
            # volume we're sampling actually has fg
            return np.random.choice(self.contains_fg_list)
        else:
            return np.random.randint(self.n_volumes)

    def __len__(self):
        return self.epoch_len * self.batch_size

    def __getitem__(self, index):

        idx = index % self.batch_size

        if idx < self.min_biased_samples:
            biased_sampling = True
        else:
            biased_sampling = np.random.rand() < self.p_bias_sampling

        ind = self._get_random_volume_ind(biased_sampling)
        volumes = self._get_volume_tuple(ind)
        shape = np.array(volumes[0].shape)[1:]

        if biased_sampling:
            # if we're not there let's choose a center coordinate
            # that contains fg
            if self.store_coords_in_ram:
                coords = self.coords_list[ind]
            else:
                # or not!
                case = os.path.basename(self.vol_ds.path_dicts[ind][self.label_key])
                coords = np.load(os.path.join(self.bias_coords_fol, case))
            n_coords = coords.shape[1]
            if n_coords > 0:
                coord = coords[:, np.random.randint(n_coords)] - self.patch_size//2
            else:
                # random coordinate
                coord = np.random.randint(np.maximum(shape - self.patch_size+1, 1))
        else:
            # random coordinate
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

        if self.is_cascade:
            # after augmentation we need to extend the prediction from the previous stage
            # from integer representation to one hot encoding where we can leave away the 
            # background
            if self.n_pred_classes > 1:
                im = volume[:self.n_im_channels]
                pred_fps = volume[self.n_im_channels:-1]
                seg = volume[-1:]
                pred_fps = np.concatenate([pred_fps == c] for c in range(1, self.n_pred_classes+1))
                pred_fps = pred_fps.astype(im.dtype)
                volume = np.concatenate([im, pred_fps, seg])

        if self.return_fp16:
            volume = volume.astype(np.float16)
        else:
            volume = volume.astype(np.float32)

        return volume


def SegmentationDataloader(vol_ds, patch_size, batch_size, num_workers=None,
                           pin_memory=True, epoch_len=250, p_bias_sampling=0,
                           min_biased_samples=1, augmentation=None, padded_patch_size=None,
                           store_coords_in_ram=True, memmap='r', n_im_channels: int = 1,
                           image_key='image',
                           label_key='label', pred_fps_key=None, n_pred_classes=None,
                           store_data_in_ram=False,
                           return_fp16=True,
                           n_max_volumes=None,
                           bias='fg'):
    dataset = SegmentationBatchDataset(vol_ds=vol_ds, patch_size=patch_size, batch_size=batch_size,
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
                                       bias=bias)
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
