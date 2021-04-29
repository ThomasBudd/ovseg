import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage.filters import gaussian_filter
from ovseg.utils.torch_np_utils import check_type


class SlidingWindowPrediction(object):

    def __init__(self, network, patch_size, batch_size=1, overlap=0.5, fp32=False,
                 patch_weight_type='gaussian', sigma_gaussian_weight=1/8, linear_min=0.1,
                 mode='flip', TTA=None, TTA_n_full_predictions=1, TTA_n_max_augs=99,
                 TTA_eps_stop=0.02):

        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network = network.to(self.dev)
        self.patch_size = np.array(patch_size)
        self.batch_size = batch_size
        self.overlap = overlap
        self.fp32 = fp32
        self.patch_weight_type = patch_weight_type
        self.sigma_gaussian_weight = sigma_gaussian_weight
        self.linear_min = linear_min
        self.mode = mode
        self.TTA = TTA
        self.TTA_n_full_predictions = TTA_n_full_predictions
        self.TTA_n_max_augs = TTA_n_max_augs
        self.TTA_eps_stop = TTA_eps_stop

        assert self.patch_weight_type.lower() in ['constant', 'gaussian', 'linear']

        # check and build up the patch weight
        # we can use a gaussian weighting since the predictions on the edge of the patch are less
        # reliable than the ones in the middle
        if self.patch_weight_type.lower() == 'constant':
            self.patch_weight = np.ones(self.patch_size)
        elif self.patch_weight_type.lower() == 'gaussian':
            # we distrust the edge voxel the same in each direction regardless of the
            # patch size in that dimension

            # thanks to Fabian Isensee! I took this from his code:
            # https://github.com/MIC-DKFZ/nnUNet/blob/14992342919e63e4916c038b6dc2b050e2c62e3c/nnunet/network_architecture/neural_network.py#L250
            tmp = np.zeros(self.patch_size)
            center_coords = [i // 2 for i in self.patch_size]
            sigmas = [i * self.sigma_gaussian_weight for i in self.patch_size]
            tmp[tuple(center_coords)] = 1
            self.patch_weight = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
            self.patch_weight = self.patch_weight / np.max(self.patch_weight) * 1
            self.patch_weight = self.patch_weight.astype(np.float32)

            # self.patch_weight cannot be 0, otherwise we may end up with nans!
            self.patch_weight[self.patch_weight == 0] = np.min(
                self.patch_weight[self.patch_weight != 0])

        elif self.patch_weight_type.lower() == 'linear':
            lin_slopes = [np.linspace(self.linear_min, 1, s//2) for s in self.patch_size]
            hats = [np.concatenate([lin_slope, lin_slope[::-1]]) for lin_slope in lin_slopes]
            hats = [np.expand_dims(hat, [j for j in range(len(self.patch_size)) if j != i])
                    for i, hat in enumerate(hats)]

            self.patch_weight = np.ones(self.patch_size)
            for hat in hats:
                self.patch_weight *= hat

        self.patch_weight = self.patch_weight[np.newaxis]
        self.patch_weight = torch.from_numpy(self.patch_weight).to(self.dev).type(torch.float)

        # add an axis to the patch size and set is_2d
        if len(self.patch_size) == 2:
            self.is_2d = True
            self.patch_size = np.concatenate([[1], self.patch_size])
        elif len(self.patch_size) == 3:
            self.is_2d = False
        else:
            raise ValueError('patch_size must be of len 2 or 3 (for 2d and 3d networks).')

    def _sliding_window(self, volume, ROI=None):

        if not torch.is_tensor(volume):
            raise TypeError('Input must be torch tensor')
        if not len(volume.shape) == 4:
            raise ValueError('Volume must be a 4d tensor (incl channel axis)')

        # in case the volume is smaller than the patch size we pad it
        # and save the input size to crop again before returning
        shape_in = np.array(volume.shape)

        # %% possible padding of too small volumes
        pad = [0, self.patch_size[2] - shape_in[3], 0, self.patch_size[1] - shape_in[2],
               0, self.patch_size[0] - shape_in[1]]
        pad = np.maximum(pad, 0).tolist()
        volume = F.pad(volume, pad).type(torch.float)
        nz, nx, ny = volume.shape[1:]

        # %% reserve storage
        pred = torch.zeros((self.network.out_channels, nz, nx, ny),
                           device=self.dev,
                           dtype=torch.float)
        ovlp = torch.zeros((1, nz, nx, ny),
                           device=self.dev,
                           dtype=torch.float)
        if ROI is None:
            # if the ROI
            ROI = torch.ones((nz, nx, ny)) > 0

        n_patches = np.ceil((np.array([nz, nx, ny]) - self.patch_size) / 
                            (self.overlap * self.patch_size)).astype(int) + 1

        # upper left corners of all patches
        z_list = np.linspace(0, nz - self.patch_size[0], n_patches[0]).astype(int).tolist()
        x_list = np.linspace(0, nx - self.patch_size[1], n_patches[1]).astype(int).tolist()
        y_list = np.linspace(0, ny - self.patch_size[2], n_patches[2]).astype(int).tolist()

        zxy_list = []
        for z in z_list:
            for x in x_list:
                for y in y_list:
                    # we only predict the patch if we intersect the ROI
                    if ROI[z:z+self.patch_size[0], x:x+self.patch_size[1],
                           y:y+self.patch_size[2]].any().item():
                        zxy_list.append((z, x, y))

        # introduce batch size
        n_full_batches = len(zxy_list) // self.batch_size
        zxy_batched = [zxy_list[i * self.batch_size: (i + 1) * self.batch_size]
                       for i in range(n_full_batches)]
        if n_full_batches * self.batch_size < len(zxy_list):
            zxy_batched.append(zxy_list[n_full_batches * self.batch_size:])

        # %% now the magic!
        with torch.no_grad():
            for zxy_batch in zxy_batched:
                # crop
                batch = torch.stack([volume[:,
                                            z:z+self.patch_size[0],
                                            x:x+self.patch_size[1],
                                            y:y+self.patch_size[2]] for z, x, y in zxy_batch])

                # remove z axis if we have 2d prediction
                batch = batch[:, :, 0] if self.is_2d else batch
                # remember that the network is outputting a list of predictions for each scale
                if not self.fp32 and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        out = self.network(batch)[0]
                else:
                    out = self.network(batch)[0]

                # add z axis again maybe
                out = out.unsqueeze(2) if self.is_2d else out

                # update pred and overlap
                for i, (z, x, y) in enumerate(zxy_batch):
                    pred[:, z:z+self.patch_size[0], x:x+self.patch_size[1],
                         y:y+self.patch_size[2]] += F.softmax(out[i], 0) * self.patch_weight
                    ovlp[:, z:z+self.patch_size[0], x:x+self.patch_size[1],
                         y:y+self.patch_size[2]] += self.patch_weight

        # %% bring maybe back to old shape
        pred = pred[:, :shape_in[1], :shape_in[2], :shape_in[3]]
        ovlp = ovlp[:, :shape_in[1], :shape_in[2], :shape_in[3]]

        # just to be sure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred / ovlp

    def predict_volume(self, volume, mode=None):
        # evaluates the siliding window on this volume
        # predictions are returned as soft segmentations
        if mode is None:
            mode = self.mode

        self.network.eval()

        # check the type and bring to device
        is_np, _ = check_type(volume)
        if is_np:
            volume = torch.from_numpy(volume).to(self.dev)

        # check if inpt is 3d or 4d for the output
        if len(volume.shape) == 3:
            volume = volume.unsqueeze(0)

        if mode == 'simple':
            pred = self._predict_volume_simple(volume)
        elif mode == 'flip':
            pred = self._predict_volume_flip(volume)
        elif mode == 'TTA':
            raise NotImplementedError('Test time augmentations were not implemented beyond '
                                      'flipping.')
            # pred = self._predict_volume_tta(volume)

        if is_np:
            pred = pred.cpu().numpy()

        return pred

    def __call__(self, volume, mode=None):
        return self.predict_volume(volume, mode)

    def _predict_volume_simple(self, volume):
        return self._sliding_window(volume)

    def _predict_volume_flip(self, volume):

        flip_z_list = [False] if self.is_2d else [False, True]

        # collect all combinations of flipping
        flip_list = []
        for fz in flip_z_list:
            for fx in [False, True]:
                for fy in [False, True]:
                    flip_list.append((fz, fx, fy))

        # do the first one outside the loop for initialisation
        pred = self._sliding_window(volume)

        # now some flippings!
        for f in flip_list[1:]:
            volume = self._flip_volume(volume, f)

            # predict flipped volume
            pred_flipped = self._sliding_window(volume)

            # flip back and update
            pred += self._flip_volume(pred_flipped, f)
            volume = self._flip_volume(volume, f)

        return pred / len(flip_list)

    def _flip_volume(self, volume, f):
        for i in range(3):
            if f[i]:
                volume = volume.flip(i+1)
        return volume

    # def _predict_volume_tta(self, volume):

    #     if self.TTA is None:
    #         raise TypeError('When test time augmentations are used the argument TTA of '
    #                         'SlidingWindowPrediction must be initialised.')

    #     # counter for the amount of augmentations we do to the image
    #     self.augs = 0
    #     # first prediction without augmentation
    #     pred = self._sliding_window(volume)

    #     # now we do some full predictions
    #     for _ in range(1, self.TTA_n_full_predictions):
    #         volume_aug = self.TTA.augment_volume(volume, is_inverse=False)
    #         pred_aug = self._sliding_window(volume_aug)
    #         pred = self.TTA.augment_volume(pred_aug, is_inverse=True)
    #         self.augs += 1

    #     # as the full predictions are over we only predict where we have error left
    #     prev_pred = torch.zeros_like(pred)

    #     # the ROI is defined as the pixels where the soft probabilities deviated
    #     # more than eps from the previous prabilities
    #     ROI = torch.abs(pred - prev_pred).sum(0) > self.TTA_eps_stop
    #     eps = ROI.max().item()
    #     while self.augs < self.TTA_n_max_augs and eps > self.TTA_eps_stop:

    #         # store current pred and ovlp
    #         prev_pred = pred

    #         # create new augmentation
    #         volume_aug = self.TTA.augment_volume(volume, is_inverse=False)
    #         self.augs += 1

    #         # do sliding window only in the ROI and update
    #         pred_aug = self._sliding_window(volume_aug, ROI)
    #         pred = pred + self.TTA.augment_volume(pred_aug, is_inverse=True)

    #         # compute new ROI and max error
    #         ROI = torch.abs(pred - prev_pred).sum(0)
    #         eps = ROI.max().item()
    #         ROI = ROI > self.TTA_eps_stop

    #     return pred
