import numpy as np
import torch
from ovseg.utils.torch_np_utils import check_type, stack
from ovseg.utils.grid_utils import rotate_grid_2d, rotate_grid_3d, \
    get_centred_np_grid, get_centred_torch_grid, scale_grid, grid_to_indices
from ovseg.utils.interp_utils import interp_sample


class SpatialAugmentation(object):
    '''
    SpatialAugmentation(patch_size, p_scale=0.2, scale_mm=[0.7, 1.4], p_rot=0.2
                        rot_mm=[-15, 15], spatial_aug_3d=False, p_flip=0.5,
                        spacing=None)

    Performs scalings, rotations and mirroring for samples and batches
    as well as for full volumes for TTA.

    Parameters:
    ----------------
    patch_size :
        size of the patches after augmentation not that you can input
        patches larger than patch_size to avoid extrapolation
    p_scale, p_rot, p_flip :
        chance to apply scaling, rotation and flipping (per axes)
    scale_mm, rot_mm :
        min and max values of the uniform distr. when drawing parameters
        for the scaling and rotation (in degrees)
    spatial_aug_3d :
        if True scaling and rotations are done in 3d
    spacing :
        voxel_spacing of the input. default: [1, 1, 1]
    '''

    def __init__(self, patch_size, p_scale=0.2, scale_mm=[0.7, 1.4],
                 p_rot=0.2, rot_mm=[-15, 15], spatial_aug_3d=False,
                 p_flip=0.5, spacing=None, n_im_channels=1):
        self.patch_size = patch_size
        self.p_scale = p_scale
        self.scale_mm = scale_mm
        self.p_rot = p_rot
        self.rot_mm = rot_mm
        self.spatial_aug_3d = spatial_aug_3d
        self.p_flip = p_flip
        self.n_im_channels = n_im_channels

        self.dim_rot = 3 if self.spatial_aug_3d else None
        self.rotate_grid = rotate_grid_3d if self.spatial_aug_3d else \
            rotate_grid_2d

        if spacing is None:
            print('No spacing initialised! Using [1, 1, 1]')
            self.spacing = np.array([1, 1, 1])
        else:
            self.spacing = np.array(spacing)

    def draw_parameters(self):
        self.do_scale_rot = False
        if np.random.rand() < self.p_scale:
            self.scale = np.random.uniform(*self.scale_mm)
            self.do_scale_rot = True
        else:
            self.scale = 1.0
        if np.random.rand() < self.p_rot:
            self.alpha = np.random.uniform(*self.rot_mm, self.dim_rot)
            self.alpha = np.deg2rad(self.alpha)
            self.do_scale_rot = True
        else:
            self.alpha = [0.0, 0.0, 0.0] if self.spatial_aug_3d else 0.0
        self.flp = np.random.rand(3) < self.p_flip

    def _flip_sample(self, sample):
        self.flp = self.flp[:len(sample.shape)-1]
        if isinstance(sample, np.ndarray):
            for i, f in enumerate(self.flp):
                if f:
                    sample = np.flip(sample, i+1)
        else:
            dims = [i+1 for i, f in enumerate(self.flp) if f]
            if len(dims) > 0:
                sample = torch.flip(sample, dims)
        return sample

    def _crop_sample(self, sample):
        shape = np.array(sample.shape[1:])
        crop = (shape - self.patch_size) // 2
        if len(shape) == 2:
            return sample[:, crop[0]: crop[0] + self.patch_size[0],
                          crop[1]: crop[1] + self.patch_size[1]]
        else:
            return sample[:, crop[0]: crop[0] + self.patch_size[0],
                          crop[1]: crop[1] + self.patch_size[1],
                          crop[2]: crop[2] + self.patch_size[2]]

    def augment_image(self, img):
        is_np, _ = check_type(img)
        if is_np:
            return self.augment_sample(img[np.newaxis])[0]
        else:
            return self.augment_sample(img.unsqueeze(0))[0]

    def augment_sample(self, sample):
        '''

        Parameters
        ----------
        sample : np.ndarray or torch.Tensor
            shape: (channels, nx, ny(, nz))

        Returns
        -------
        sample_aug : np.ndarray or torch.Tensor
            shape: (channels, *patch_size), augmented sample

        '''
        is_np, _ = check_type(sample)
        self.draw_parameters()
        if self.do_scale_rot:
            if is_np:
                if self.spatial_aug_3d:
                    grid = get_centred_np_grid(self.patch_size, self.spacing)
                else:
                    grid = get_centred_np_grid(self.patch_size[:2],
                                               self.spacing[:2])
                orders = self.n_im_channels * [3] + \
                    (len(sample)-self.n_im_channels) * [0]
            else:
                if self.spatial_aug_3d:
                    grid = get_centred_torch_grid(self.patch_size,
                                                  self.spacing,
                                                  device=sample.device)
                else:
                    grid = get_centred_torch_grid(self.patch_size[:2],
                                                  self.spacing[:2],
                                                  device=sample.device)
                orders = self.n_im_channels * [1] + \
                    (len(sample)-self.n_im_channels) * [0]
                grid = grid.to(sample.device)
            grid = scale_grid(grid, self.scale)
            grid = self.rotate_grid(grid, self.alpha)
            # the sample shape might be greater than the patch size
            # to avoid extrapolation values in the sample
            # so here we use sample.shape instead of patch_size
            grid = grid_to_indices(grid, sample.shape[1:], self.spacing)
            sample = interp_sample(sample, grid, orders)
        else:
            sample = self._crop_sample(sample)
        # now do some axes flipping
        sample = self._flip_sample(sample)
        return sample

    def augment_batch(self, batch):
        '''

        Parameters
        ----------
        batch : np.ndarray or torch.Tensor, or list of samples
            shape: (batchsize, channels, nx, ny(, nz))

        Returns
        -------
        batch_aug : np.ndarray or torch.Tensor
            shape: (batchsize, channels, *patch_size), augmented batch

        '''

        return stack([self.augment_sample(batch[i])
                      for i in range(len(batch))])

    def augment_volume(self, volume, is_inverse: bool = False):
        '''

        Parameters
        ----------
        volume : np.ndarray or torch.tensor
            full 3d volume for augmentation.
            shape: (nx, ny, nz) or (channels, nx, ny, nz)
        is_inverse : bool, optional
            if True the inverse augmentation of the last call
            of this function is applied. Usefull for TTA:
                volume_aug = augment_volume(volume)
                pred_aug = predict(volume_aug)
                pred = augment_volume(pred_aug, True)
            This way an infinite number of coregistered prediction can be
            created
        Returns
        -------
        volume : np.ndarray or toch.tensor
            augmented volume

        '''
        is_np, _ = check_type(volume)
        img_inpt = len(volume.shape) == 3
        if img_inpt:
            if is_np:
                volume = volume[np.newaxis]
            else:
                volume = volume.unsqueeze(0)
        shape_in = np.array(volume.shape)[1:]
        if is_inverse:
            # interpolate back to the old shape and
            # don't draw new parameters
            shape_new = self.shape_old
            # in the inverse we first have to flip and then to the rotation
            # and everything
            volume = self._flip_sample(volume)
        else:
            # draw new parameters and save them for the next inverse
            # augmentation
            self.draw_parameters()
            # keep input shape
            if self.spatial_aug_3d:
                shape_new = np.round(shape_in / self.scale)
                self.shape_old = shape_in
            else:
                shape_new = np.round(shape_in[:2] / self.scale)
                self.shape_old = shape_in[:2]
        # we create grids of that cover a cube of the size
        # shape_old * spacing but with shape_new grid points
        # this sub/super sampling incooperates automaticall the scaling
        if is_np:
            grid = get_centred_np_grid(self.shape_old, self.spacing, shape_new)
        else:
            grid = get_centred_torch_grid(self.shape_old, self.spacing,
                                          shape_new).to(volume.device)
        # rotate the grid
        grid = self.rotate_grid(grid, self.alpha, is_inverse=is_inverse)
        # back to indices of the original input volume
        grid = grid_to_indices(grid, self.shape_old, self.spacing)
        if is_inverse:
            # for the inverse case we have to scale the coordinates again to
            # be indices of "volume"
            grid = grid / self.scale
        # now interpolation!
        orders = len(volume) * [1]
        volume = interp_sample(volume, grid, orders)
        # and flipping!
        if not is_inverse:
            volume = self._flip_sample(volume)
        if img_inpt:
            return volume[0]
        else:
            return volume
