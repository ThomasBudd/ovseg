import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage.filters import gaussian_filter
from ovseg.utils.torch_np_utils import check_type, maybe_add_channel_dim


class SlidingWindowPrediction(object):

    def __init__(self, network, patch_size, batch_size=1, overlap=0.5, fp32=False,
                 patch_weight_type='gaussian', sigma_gaussian_weight=1/8, linear_min=0.1,
                 mode='flip'):

        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network = network.to(self.dev)
        self.batch_size = batch_size
        self.overlap = overlap
        self.fp32 = fp32
        self.patch_weight_type = patch_weight_type
        self.sigma_gaussian_weight = sigma_gaussian_weight
        self.linear_min = linear_min
        self.mode = mode

        assert self.patch_weight_type.lower() in ['constant', 'gaussian', 'linear']
        assert self.mode.lower() in ['simple', 'flip']

        self._set_patch_size_and_weight(patch_size)
    

    def _set_patch_size_and_weight(self, patch_size): 
        # check and build up the patch weight
        # we can use a gaussian weighting since the predictions on the edge of the patch are less
        # reliable than the ones in the middle
        self.patch_size = np.array(patch_size).astype(int)
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
        


    def _get_xyz_list(self, shape, ROI=None):
        
        nz, nx, ny = shape
        
        if ROI is None:
            # not ROI is given take all coordinates
            ROI = torch.ones((1, nz, nx, ny)) > 0

        n_patches = np.ceil((np.array([nz, nx, ny]) - self.patch_size) / 
                            (self.overlap * self.patch_size)).astype(int) + 1

        # upper left corners of all patches
        if self.is_2d:
            z_list = np.arange(nz).astype(int).tolist()
        else:
            z_list = np.linspace(0, nz - self.patch_size[0], n_patches[0]).astype(int).tolist()
        x_list = np.linspace(0, nx - self.patch_size[1], n_patches[1]).astype(int).tolist()
        y_list = np.linspace(0, ny - self.patch_size[2], n_patches[2]).astype(int).tolist()

        zxy_list = []
        for z in z_list:
            for x in x_list:
                for y in y_list:
                    # we only predict the patch if the middle cube with half side length
                    # intersects the ROI
                    if self.is_2d:
                        z1, z2 = z, z+1
                    else:
                        z1, z2 = z+self.patch_size[0]//4, z+self.patch_size[0]*3//4
                    x1, x2 = x+self.patch_size[1]//4, x+self.patch_size[1]*3//4
                    y1, y2 = y+self.patch_size[2]//4, y+self.patch_size[2]*3//4
                    if ROI[0, z1:z2, x1:x2, y1:y2].any().item():
                        zxy_list.append((z, x, y))

        return zxy_list

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
        shape = volume.shape[1:]

        # %% reserve storage
        pred = torch.zeros((self.network.out_channels, *shape),
                           device=self.dev,
                           dtype=torch.float)
        # this is for the voxel where we have no prediction in the end
        # for each of those the method will return the (1,0,..,0) vector
        # pred[0] = 1
        ovlp = torch.zeros((1, *shape),
                           device=self.dev,
                           dtype=torch.float)
        
        # %% get all top left coordinates of patches
        zxy_list = self._get_xyz_list(shape, ROI)
        
        # introduce batch size
        # some people say that introducing a batch size at inference time makes it faster
        # I couldn't see that so far
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

            # set the prediction to background and prevent zero division where
            # we did not evaluate the network
            pred[0, ovlp[0] == 0] = 1
            ovlp[ovlp == 0] = 1

            pred /= ovlp

            # just to be sure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
            return pred

    def predict_volume(self, volume, ROI=None, mode=None):
        # evaluates the siliding window on this volume
        # predictions are returned as soft segmentations
        if mode is None:
            mode = self.mode

        if ROI is not None:
            ROI = maybe_add_channel_dim(ROI)

        self.network.eval()

        # check the type and bring to device
        is_np, _ = check_type(volume)
        if is_np:
            volume = torch.from_numpy(volume).to(self.dev)

        # check if inpt is 3d or 4d for the output
        volume = maybe_add_channel_dim(volume)

        if mode.lower() == 'simple':
            pred = self._predict_volume_simple(volume, ROI)
        elif mode.lower() == 'flip':
            pred = self._predict_volume_flip(volume, ROI)

        if is_np:
            pred = pred.cpu().numpy()

        return pred

    def __call__(self, volume, ROI=None, mode=None):
        return self.predict_volume(volume, ROI, mode)

    def _predict_volume_simple(self, volume, ROI=None):
        return self._sliding_window(volume, ROI)

    def _predict_volume_flip(self, volume, ROI=None):

        flip_z_list = [False] if self.is_2d else [False, True]
        
        if ROI is not None and isinstance(ROI, np.ndarray):
            ROI = torch.from_numpy(ROI)

        # collect all combinations of flipping
        flip_list = []
        for fz in flip_z_list:
            for fx in [False, True]:
                for fy in [False, True]:
                    flip_list.append((fz, fx, fy))

        # do the first one outside the loop for initialisation
        pred = self._sliding_window(volume, ROI=ROI)

        # now some flippings!
        for f in flip_list[1:]:
            volume = self._flip_volume(volume, f)
            if ROI is not None:
                ROI = self._flip_volume(ROI, f)

            # predict flipped volume
            pred_flipped = self._sliding_window(volume, ROI)

            # flip back and update
            pred += self._flip_volume(pred_flipped, f)
            volume = self._flip_volume(volume, f)
            if ROI is not None:
                ROI = self._flip_volume(ROI, f)

        return pred / len(flip_list)

    def _flip_volume(self, volume, f):
        for i in range(3):
            if f[i]:
                volume = volume.flip(i+1)
        return volume
