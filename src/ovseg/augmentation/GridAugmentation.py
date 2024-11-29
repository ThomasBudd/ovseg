import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class torch_inplane_grid_augmentations(nn.Module):

    def __init__(self,
                 p_rot=0.2,
                 p_zoom=0.2,
                 p_scale_if_zoom=0,
                 p_transl=0,
                 p_shear=0,
                 mm_zoom=[0.7, 1.4],
                 mm_rot=[-15, 15],
                 mm_transl=[-0.25, 0.25],
                 mm_shear=[-0.2, 0.2],
                 apply_flipping=True,
                 n_im_channels: int = 1,
                 out_shape=None
                 ):
        super().__init__()
        self.p_rot = p_rot
        self.p_zoom = p_zoom
        self.p_scale_if_zoom = p_scale_if_zoom
        self.p_transl = p_transl
        self.p_shear = p_shear
        self.mm_zoom = mm_zoom
        self.mm_rot = mm_rot
        self.mm_transl = mm_transl
        self.mm_shear = mm_shear
        self.apply_flipping = apply_flipping
        self.n_im_channels = n_im_channels
        self.out_shape = out_shape
        if out_shape is not None:
            self.out_shape = np.array(self.out_shape)

    def _rot(self, theta):

        angle = np.random.uniform(*self.mm_rot)
        rot_m = torch.zeros_like(theta[:, :-1])
        cos, sin = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        rot_m[0, 0] = cos
        rot_m[0, 1] = sin
        rot_m[1, 0] = -1 * sin
        rot_m[1, 1] = cos
        if theta.shape[0] == 3:
            rot_m[2, 2] = 1

        return torch.mm(rot_m, theta)

    def _zoom(self, theta):
        fac1 = np.random.uniform(*self.mm_zoom)
        if np.random.rand() < self.p_scale_if_zoom:
            fac2 = np.random.uniform(*self.mm_zoom)
        else:
            fac2 = fac1
        theta[0, 0] *= fac1
        theta[1, 1] *= fac2
        theta[0, -1] *= fac1
        theta[1, -1] *= fac2
        return theta

    def _translate(self, theta):
        theta[0, -1] = np.random.uniform(*self.mm_transl)
        theta[1, -1] = np.random.uniform(*self.mm_transl)
        return theta

    def _shear(self, theta):
        s = np.random.uniform(*self.mm_shear)
        shear_m = torch.zeros_like(theta[:, :-1])
        for i in range(theta.shape[0]):
            shear_m[i, i] = 1
        if np.random.rand() < 0.5:
            shear_m[0, 1] = s
        else:
            shear_m[1, 0] = s
        return torch.mm(shear_m, theta)

    def _get_ops_list(self):
        ops_list = []
        if np.random.rand() < self.p_rot:
            ops_list.append(self._rot)
        if np.random.rand() < self.p_zoom:
            ops_list.append(self._zoom)
        if np.random.rand() < self.p_transl:
            ops_list.append(self._translate)
        if np.random.rand() < self.p_shear:
            ops_list.append(self._shear)
        np.random.shuffle(ops_list)

        return ops_list

    def _flip(self, xb):

        bs = xb.shape[0]
        img_dims = len(xb.shape) - 2
        flp_list = [np.random.rand(img_dims) < 0.5 for _ in range(bs)]

        for b, flp in enumerate(flp_list):
            dims = [i + 1 for i, f in enumerate(flp) if f]
            if len(dims) > 0:
                xb[b] = torch.flip(xb[b], dims)
        return xb

    def forward(self, xb):

        bs, n_ch = xb.shape[0:2]
        img_dims = len(xb.shape) - 2
        theta = torch.zeros((bs, img_dims, img_dims+1), device=xb.device, dtype=xb.dtype)
        for j in range(img_dims):
            theta[:, j, j] = 1
        for i in range(bs):
            ops_list = self._get_ops_list()
            for op in ops_list:
                theta[i] = op(theta[i])

        grid = F.affine_grid(theta, xb.size()).cuda().type(xb.dtype)
        if self.out_shape is not None:
            # crop from the grid
            crp_l = (np.array(xb.shape[2:]) - self.out_shape) // 2
            crp_u = (crp_l + self.out_shape)
            grid = grid[:, crp_l[0]:crp_u[0], crp_l[1]:crp_u[1], crp_l[2]:crp_u[2]]
        xb = torch.cat([F.grid_sample(xb[:, :self.n_im_channels], grid, mode='bilinear'),
                        F.grid_sample(xb[:, self.n_im_channels:], grid, mode='nearest')], dim=1)

        # now flipping
        if self.apply_flipping:
            xb = self._flip(xb)
        return xb

    def update_prg_trn(self, param_dict, h, indx=None):

        attr_list = ['p_rot', 'p_zoom', 'p_transl', 'p_shear', 'mm_zoom', 'mm_rot',
                     'mm_transl', 'mm_shear']

        for attr in attr_list:
            if attr in param_dict:
                self.__setattr__(attr, (1 - h) * param_dict[attr][0] + h * param_dict[attr][1])

        if 'out_shape' in param_dict:
            self.out_shape = param_dict['out_shape'][indx]

# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from time import perf_counter
    plt.close('all')
    
    im_full = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04_test\\default\\images'
                      '\\OV04_034_20091014.npy')
    lb_full = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04_test\\default\\labels'
                      '\\OV04_034_20091014.npy') > 0
    
    im_crop = im_full[30:78, 100:292, 100:292].astype(np.float32)
    imt = torch.from_numpy(im_crop).cuda().unsqueeze(0).unsqueeze(0).type(torch.float)
    lb_crop = lb_full[30:78, 100:292, 100:292].astype(np.float32)
    lbt = torch.from_numpy(lb_crop).cuda().unsqueeze(0).unsqueeze(0).type(torch.float)
    xb = torch.cat([imt, lbt], 1).cuda()
    xb = torch.cat([xb, xb], 0)
    aug = torch_inplane_grid_augmentations(p_rot=0.5, p_zoom=0.5, p_scale_if_zoom=0.5,
                                           p_transl=0.0, p_shear=0.5,
                                           mm_zoom=[0.8,1.2], mm_rot=[-20, 20],
                                           apply_flipping=False)

    # %%
    xb_aug = aug(xb).cpu().numpy()

    z = np.argmax(np.sum(lb_crop > 0, (1, 2)))
    plt.subplot(1, 3, 1)
    plt.imshow(xb_aug[0, 0, z], cmap='gray')
    plt.contour(xb_aug[0, 1, z])
    plt.subplot(1, 3, 2)
    plt.imshow(im_crop[z], cmap='gray')
    plt.contour(lb_crop[z])
    plt.subplot(1, 3, 3)
    plt.imshow(xb_aug[1, 0, z], cmap='gray')
    plt.contour(xb_aug[1, 1, z])

    # %%

    st = perf_counter()
    for _ in range(50):
        xb_aug = aug(xb)
    torch.cuda.synchronize()
    et = perf_counter()
    print('It took {:.7f}s for augmenting with batch size 2'.format((et-st)/50))