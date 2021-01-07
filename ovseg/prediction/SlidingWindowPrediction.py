import numpy as np
import torch
from torch.nn import functional as F
from ovseg.utils.torch_np_utils import check_type


class SlidingWindowPrediction(object):

    def __init__(self, network, patch_size, batch_size=1, overlap=0.5, fp32=False,
                 patch_weight_type='constant', sigma_gaussian_weight=1, TTA=None,
                 mode='simple', TTA_n_full_predictions=1, TTA_n_max_augs=99,
                 TTA_eps_stop=0.02):

        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network = network.to(self.dev)
        self.patch_size = np.array(patch_size)
        self.batch_size = batch_size
        self.overlap = overlap
        self.fp32 = fp32
        self.patch_weight_type = patch_weight_type
        self.sigma_gaussian_weight = sigma_gaussian_weight
        self.TTA = TTA
        self.mode = mode
        self.TTA_n_full_predictions = TTA_n_full_predictions
        self.TTA_n_max_augs = TTA_n_max_augs
        self.TTA_eps_stop = TTA_eps_stop

        if len(self.patch_size) == 2:
            self.is_2d = True
            self.patch_size = np.concatenate([self.patch_size, [1]])
        elif len(self.patch_size) == 3:
            self.is_2d = False
        else:
            raise ValueError('patch_size must be of len 2 or 3 (for 2d and 3d networks).')

        assert self.patch_weight_type.lower() in ['constant', 'gaussian']

        # check and build up the patch weight
        # we can use a gaussian weighting since the predictions on the edge of the patch are less
        # reliable than the ones in the middle
        if self.patch_weight_type.lower() == 'constant':
            self.patch_weight = np.ones(self.patch_size)
        elif self.patch_weight_type.lower() == 'gaussian':
            # we distrust the edge voxel the same in each direction regardless of the
            # patch size in that dimension
            axes = [np.linspace(-1, 1, p) for p in self.patch_size]

            if self.is_2d:
                # linspace(-1, 1, 1) = [-1]
                axes[2] = [0]

            grid = np.stack(np.meshgrid(axes))
            norm_sq = np.sum(grid**2, 0)
            self.patch_weight = np.exp(-0.5 * norm_sq/self.sigma_gaussian_weight**2)
            self.patch_weight = self.patch_weight.astype(np.float32)
            n_zeros = np.sum(self.patch_weight == 0)
            if n_zeros > 0:
                print('Small sigma for gaussian weighting. {:.3f} % of the weights are 0 and will '
                      'be set to 1e-5.'.format(100*n_zeros/self.patch_weight.size))
                self.patch_weight = np.maximum(self.patch_weight, 1e-5)
        else:
            raise ValueError('Unkown patch_weight_type {}. Known types: [constant, gaussian]'
                             ''.format(self.patch_weight_type))

        self.patch_weight = torch.from_numpy(self.patch_weight).to(self.dev)

    def _sliding_window(self, volume, ROI=None):

        if not torch.is_tensor(volume):
            raise TypeError('Input must be torch tensor')
        if not len(volume.shape) == 4:
            raise ValueError('Volume must be a 4d tensor (incl channel axis)')

        # in case the volume is smaller than the patch size we pad it
        # and save the input size to crop again before returning
        shape_in = volume.shape

        # %% possible padding of too small volumes
        pad = [0, self.patch_size[2] - shape_in[3], 0, self.patch_size[1] - shape_in[2],
               0, self.patch_size[0] - shape_in[1]]
        pad = np.maximum(pad, 0).tolist()
        volume = F.pad(volume, pad)
        nx, ny, nz = volume.shape[1:]

        # %% reserve storage
        pred = torch.zeros((self.network.out_channels, nx, ny, nz), device=self.dev)
        ovlp = torch.zeros((1, nx, ny, nz), device=self.dev)
        if ROI is None:
            # if the ROI
            ROI = torch.ones((nx, ny, nz)) > 0

        # upper left corners of all patches
        x_list = list(range(0, nx - self.patch_size[0],
                            int(self.patch_size[0] * self.overlap))) \
            + [nx - self.patch_size[0]]
        y_list = list(range(0, ny - self.patch_size[1],
                            int(self.patch_size[1] * self.overlap))) \
            + [ny - self.patch_size[1]]
        z_list = list(range(0, nz - self.patch_size[2],
                            max([int(self.patch_size[2] * self.overlap), 1]))) \
            + [nz - self.patch_size[2]]
        xyz_list = []
        for x in x_list:
            for y in y_list:
                for z in z_list:
                    # we only predict the patch if we intersect the ROI
                    if ROI[x:x+self.patch_size[0], y:y+self.patch_size[1],
                           z:z+self.patch_size[2]].any().item():
                        xyz_list.append((x, y, z))

        # introduce batch size
        n_full_batches = len(xyz_list) // self.batch_size
        xyz_batched = [xyz_list[i * self.batch_size: (i + 1) * self.batch_size]
                       for i in range(n_full_batches)]
        if n_full_batches * self.batch_size < len(xyz_list):
            xyz_batched.append(xyz_list[n_full_batches * self.batch_size:])

        # %% now the magic!
        with torch.no_grad():
            for xyz_batch in xyz_batched:
                # crop
                batch = torch.stack([volume[:, x:x+self.patch_size[0], y:y+self.patch_size[1],
                                            z:z+self.patch_size[2]] for x, y, z in xyz_batch])

                # remove z axis if we have 2d prediction
                batch = batch[..., 0] if self.is_2d else batch
                # remember that the network is outputting a list of predictions for each scale
                if not self.fp32 and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        out = self.network(batch)[0]
                else:
                    out = self.network(batch)[0]

                # add z axis again maybe
                out = out.unsqueeze(-1) if self.is_2d else out

                # update pred and overlap
                for i, (x, y, z) in enumerate(xyz_batch):
                    pred[:, x:x+self.patch_size[0], y:y+self.patch_size[1],
                         z:z+self.patch_size[2]] += out[i] * self.patch_weight
                    ovlp[:, x:x+self.patch_size[0], y:y+self.patch_size[1],
                         z:z+self.patch_size[2]] += self.patch_weight

        # %% bring maybe back to old shape
        pred = pred[:, :shape_in[1], :shape_in[2], :shape_in[3]]
        ovlp = ovlp[:, :shape_in[1], :shape_in[2], :shape_in[3]]

        # just to be sure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # note that the network returns logits
        return F.softmax(pred, 0), ovlp

    def predict_volume(self, volume, mode=None):
        # evaluates the siliding window on this volume
        # predictions are returned as soft segmentations
        if mode is None:
            mode = self.mode

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
            pred = self._predict_volume_tta(volume)

        if is_np:
            pred = pred.cpu().numpy()

        return pred

    def __call__(self, volume, mode=None):
        return self.predict_volume(volume, mode)

    def _predict_volume_simple(self, volume):
        pred, ovlp = self._sliding_window(volume)
        return pred/ovlp

    def _predict_volume_flip(self, volume):

        flip_z_list = [False] if self.is_2d else [False, True]

        # collect all combinations of flipping
        flip_list = []
        for fx in [False, True]:
            for fy in [False, True]:
                for fz in flip_z_list:
                    flip_list.append((fx, fy, fz))

        # do the first one outside the loop for initialisation
        pred, ovlp = self._sliding_window(volume)

        # now some flippings!
        for f in flip_list[1:]:
            volume = self._flip_volume(volume, f)

            # predict flipped volume
            pred_flipped, ovlp_flipped = self._sliding_window(volume)

            # flip back and update
            pred += self._flip_volume(pred_flipped, f)
            ovlp += self._flip_volume(ovlp_flipped, f)
            volume = self._flip_volume(volume, f)

        return pred/ovlp

    def _flip_volume(self, volume, f):
        for i in range(3):
            if f[i]:
                volume = volume.flip(i+1)
        return volume

    def _predict_volume_tta(self, volume):

        if self.TTA is None:
            raise TypeError('When test time augmentations are used the argument TTA of '
                            'SlidingWindowPrediction must be initialised.')

        # counter for the amount of augmentations we do to the image
        self.augs = 0
        # first prediction without augmentation
        pred, ovlp = self._sliding_window(volume)

        # now we do some full predictions
        for _ in range(1, self.TTA_n_full_predictions):
            volume_aug = self.TTA.augment_volume(volume, is_inverse=False)
            pred_aug, ovlp_aug = self._sliding_window(volume_aug)
            pred = self.TTA.augment_volume(pred_aug, is_inverse=True)
            ovlp = self.TTA.augment_volume(ovlp_aug, is_inverse=True)
            self.augs += 1

        # as the full predictions are over we only predict where we have error left
        prev_pred = torch.zeros_like(pred)
        prev_ovlp = torch.ones_like(ovlp)

        # the ROI is defined as the pixels where the soft probabilities deviated
        # more than eps from the previous prabilities
        ROI = torch.abs(pred / ovlp - prev_pred / prev_ovlp).sum(0) > self.TTA_eps_stop
        eps = ROI.max().item()
        while self.augs < self.TTA_n_max_augs and eps > self.TTA_eps_stop:

            # store current pred and ovlp
            prev_pred, prev_ovlp = pred, ovlp

            # create new augmentation
            volume_aug = self.TTA.augment_volume(volume, is_inverse=False)
            self.augs += 1

            # do sliding window only in the ROI and update
            pred_aug, ovlp_aug = self._sliding_window(volume_aug, ROI)
            pred = pred + self.TTA.augment_volume(pred_aug, is_inverse=True)
            ovlp = ovlp + self.TTA.augment_volume(ovlp_aug, is_inverse=True)

            # compute new ROI and max error
            ROI = torch.abs(pred / ovlp - prev_pred / prev_ovlp).sum(0)
            eps = ROI.max().item()
            ROI = ROI > self.TTA_eps_stop

        return pred / ovlp
