import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class torch_grid_augmentation(nn.Module):

    def __init__(self,
                 p_rot=0.2,
                 p_zoom=0.2,
                 p_transl=0,
                 p_shear=0,
                 mm_zoom=[0.7, 1.4],
                 mm_rot=[-15, 15],
                 mm_transl=[-0.25, 0.25],
                 mm_shear=[-0.1, 0.1],
                 apply_flipping=True,
                 threeD_affine=False,
                 mode='only_one_axis',
                 out_shape=None
                 ):
        super().__init__()
        self.p_rot = p_rot
        self.p_zoom = p_zoom
        self.p_transl = p_transl
        self.p_shear = p_shear
        self.mm_zoom = mm_zoom
        self.mm_rot = mm_rot
        self.mm_transl = mm_transl
        self.mm_shear = mm_shear
        self.apply_flipping = apply_flipping
        self.threeD_affine = threeD_affine
        self.mode = mode
        self.out_shape = out_shape

        if self.threeD_affine:
            raise NotImplementedError('Only standard 2d and 2.5 affine transformations '
                                      'implemented at the moment.')

    def _rot_z(self, theta, angle):

        rot_m = torch.zeros_like(theta[:, :-1])
        cos, sin = torch.cos(angle), torch.sin(angle)
        rot_m[0, 0] = cos
        rot_m[0, 1] = sin
        rot_m[1, 0] = -1 * sin
        rot_m[1, 1] = cos

        return torch.mm(rot_m, theta)

    def _zoom(self, theta, fac):
        return theta * fac

    def _get_theta(self, xb):

        bs = xb.shape[0]
        dims = len(xb.shape) - 2
        theta = torch.zeros((bs, dims, dims+1), device=xb.device, dtype=xb.dtype)
        do_aug = False
        for i in range(bs):
            if np.random.rand() < self.p_rot:
                angle = np.random.uniform(*self.mm_rot)
                theta[i] = self._rot_z(theta[i], angle)
                do_aug = True
            if np.random.rand() < self.p_zoom:
                fac = np.random.uniform(*self.mm_zoom)
                theta[i] = self._zoom(theta, fac)
                do_aug = True

        if do_aug:
            grid = F.affine_grid(theta, imt.size()).cuda()
            im_trsf = F.grid_sample(imt, grid).cpu().numpy()
            xb_aug = 