import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
import numpy as np
import random


# %%
class torch_myRandAugment(torch.nn.Module):
    '''
    This is really just the nnU-Net gray value Augmentation but parametrised differently
    '''

    def __init__(self,
                 P=0.15,
                 M=15,
                 n_im_channels: int = 1
                 ):
        super().__init__()
        self.P = P
        self.M = M
        self.n_im_channels = n_im_channels

    def _uniform(self, mm, device='cpu'):
        return (mm[1] - mm[0]) * torch.rand([], device=device) + mm[0]

    def _sign(self):
        return np.random.choice([-1, 1])

    def _noise(self, img, m):
        var = self._uniform([0, 0.1*m/15], device=img.device)
        sigma = torch.sqrt(var)
        return img + sigma * torch.randn_like(img)

    def _blur(self, img, m):
        sigma = self._uniform([0.5 * m/15, 0.5 + m/15], device=img.device)
        var = sigma ** 2
        axes = torch.arange(-5, 6, device=img.device)
        grid = torch.stack(torch.meshgrid([axes for _ in range(2)]))
        gkernel = torch.exp(-1*torch.sum(grid**2, dim=0)/2.0/var)
        gkernel = gkernel/gkernel.sum()
        if len(img.shape) == 4:
            # 2d case
            gkernel = gkernel.view(1, 1, 11, 11).to(img.device).type(img.dtype)
            return torch.nn.functional.conv2d(img, gkernel, padding=5)
        else:
            gkernel = gkernel.view(1, 1, 1, 11, 11).to(img.device).type(img.dtype)
            return torch.nn.functional.conv3d(img, gkernel, padding=(0, 5, 5))

    def _brightness(self, img, m):
        fac = self._uniform([1 - 0.3 * m/15, 1 + 0.3 * m/15], device=img.device)
        return img * fac

    def _contrast(self, img, m):
        fac = self._uniform([1 - 0.45 * m/15, 1 + 0.5 * m/15], device=img.device)
        mean = img.mean()
        mn = img.min().item()
        mx = img.max().item()
        img = (img - mean) * fac + mean
        return img.clamp(mn, mx)

    def _low_res(self, img, m):
        size = img.size()[2:]
        mode = 'bilinear' if len(size) == 2 else 'trilinear'
        fac = np.random.uniform(*[1, 1 + m/15])
        img = interpolate(img, scale_factor=1/fac)
        return interpolate(img, size=size, mode=mode)

    def _gamma(self, img, m):
        with torch.cuda.amp.autocast(enabled=False):
            mn, mx = img.min(), img.max()
            img = (img - mn)/(mx - mn)
            gamma = np.random.uniform(*[1 - 0.3 * m/15, 1 + 0.5 * m/15])
            if np.random.rand() < self.P:
                img = 1 - (1 - img) ** gamma
            else:
                img = img ** gamma

            return (mx - mn) * img + mn

    def _get_ops_mag_list(self):
        ops_mag_list = []
        if np.random.rand() < self.P:
            ops_mag_list.append((self._noise, np.random.rand() * self.M))
        if np.random.rand() < self.P:
            ops_mag_list.append((self._blur, np.random.rand() * self.M))
        if np.random.rand() < self.P:
            ops_mag_list.append((self._brightness, np.random.rand() * self.M))
        if np.random.rand() < self.P:
            ops_mag_list.append((self._contrast, np.random.rand() * self.M))
        if np.random.rand() < self.P:
            ops_mag_list.append((self._low_res, np.random.rand() * self.M))
        np.random.shuffle(ops_mag_list)

        return ops_mag_list

    def forward(self, xb):

        c = self.n_im_channels

        for b in range(xb.shape[0]):
            for op, m in self._get_ops_mag_list():
                xb[b:b+1, :c] = op(xb[b:b+1, :c], m)
        return xb

    def update_prg_trn(self, param_dict, h, indx=None):

        if 'M' in param_dict:
            self.M = (1 - h) * param_dict['M'][0] + h * param_dict['M'][1]

        if 'P' in param_dict:
            self.P = (1 - h) * param_dict['P'][0] + h * param_dict['P'][1]


# %%
class torch_myRandAugment_old(nn.Module):

    def __init__(self, n, m, n_im_channels=1, use_3d_spatials=False):
        super().__init__()
        self.n = n
        self.m = m
        self.n_im_channels = n_im_channels
        self.use_3d_spatials = use_3d_spatials

        # smooth_kernel = [[1, 1, 1], [1, 15, 1], [1, 1, 1]]
        smooth_kernel = [[1, 1, 1, 1, 1], [1, 5, 5, 5, 1], [1, 5, 44, 5, 1],
          [1, 5, 5, 5, 1], [1, 1, 1, 1, 1]]
        smooth_kernel = torch.tensor(smooth_kernel).type(torch.float)
        smooth_kernel = smooth_kernel / smooth_kernel.sum()
        self.smooth_kernel_2d = smooth_kernel.unsqueeze(0).unsqueeze(0)
        self.smooth_kernel_3d = self.smooth_kernel_2d.unsqueeze(0)
        self.padding_2d = (2, 2)
        self.padding_3d = (0, 2, 2)

        self.all_ops = [(self._identity, 0, 1),
                        (self._translate_x, 0, 0.33),
                        (self._translate_y, 0, 0.33),
                        (self._shear_x, 0, 0.3),
                        (self._shear_y, 0, 0.3),
                        (self._contrast, 0, 0.9),
                        (self._brightness, 0, 0.9),
                        (self._darkness, 0, 0.9),
                        #self._narrow_window,
                        (self._sharpness, 0, 0.9),
                        (self._noise, 0, 1.0)]

    def _get_theta_id(self, img):
        # helper to create the identity matrix for spatial operations
        bs, n_ch = img.shape[0:2]
        img_dims = len(img.shape) - 2
        theta = torch.zeros((bs, img_dims, img_dims+1), device=img.device, dtype=img.dtype)
        for j in range(img_dims):
            theta[:, j, j] = 1
        return theta

    def _interp_img(self, img, theta):
        # performs spatial operations by interpolation
        grid = F.affine_grid(theta, img.size()).to(img.device).type(img.dtype)
        img = torch.cat([F.grid_sample(img[:, :self.n_im_channels], grid, mode='bilinear'),
                         F.grid_sample(img[:, self.n_im_channels:], grid, mode='nearest')], dim=1)
        return img

    def _sign(self):
        return np.random.choice([-1, 1])

    # list of all transformations we take into account
    def _identity(self, img, val):
        return img

    def _translate_x(self, img, val):
        theta = self._get_theta_id(img)
        theta[:, 1, -1] = self._sign() * val
        return self._interp_img(img, theta)

    def _translate_y(self, img, val):
        theta = self._get_theta_id(img)
        theta[:, 0, -1] = self._sign() * val
        return self._interp_img(img, theta)

    def _shear_x(self, img, val):
        theta = self._get_theta_id(img)
        theta[:, 0, 1] = val * self._sign()
        return self._interp_img(img, theta)

    def _shear_y(self, img, val):
        theta = self._get_theta_id(img)
        theta[:, 1, 0] = val * self._sign()
        return self._interp_img(img, theta)

    def _contrast(self, img, val):
        val = val * self._sign()
        for ch in range(self.n_im_channels):
            img[:, ch] = (1 - val) * img[:, ch] + val * img[:, ch].mean()
        return img

    def _brightness(self, img, val):
        val = val * self._sign()
        for ch in range(self.n_im_channels):
            img[:, ch] = (1 - val) * img[:, ch] + val * img[:, ch].min()
        return img

    def _darkness(self, img, val):
        val = val * self._sign()
        for ch in range(self.n_im_channels):
            img[:, ch] = (1 - val) * img[:, ch] + val * img[:, ch].max()
        return img

    def _narrow_window(self, img, val):
        
        for ch in range(self.n_im_channels):
            mn, mx = img[:, ch].min(), img[:, ch].max()
            mn_new = mn * (1 - val) + val * mx
            mx_new = mn * val + (1 - val) * mx
            img[:, ch] = img[:, ch].clip(mn_new, mx_new)
        return img

    def _sharpness(self, img, val):
        val = val * self._sign()
        if len(img.shape) == 4:
            img_smooth = [F.conv2d(img[:, ch:ch+1],
                                   self.smooth_kernel_2d.to(img.device).type(img.dtype),
                                   padding=self.padding_2d) for ch in range(self.n_im_channels)]
        else:
            img_smooth = [F.conv3d(img[:, ch:ch+1],
                                   self.smooth_kernel_3d.to(img.device).type(img.dtype),
                                   padding=self.padding_3d) for ch in range(self.n_im_channels)]
        img_smooth = torch.cat(img_smooth, 1)
        for ch in range(self.n_im_channels):
            img[:, ch] = img_smooth[:, ch] * val + (1 - val) * img[:, ch]
        return img

    def _noise(self, img, val):
        for ch in range(self.n_im_channels):
            img[:, ch] = img[:, ch] + val * torch.randn_like(img[:, ch])
        return img

    def forward(self, xb):

        for b in range(xb.shape[0]):
            ops_list = random.choices(self.all_ops, k=self.n)
            for op, mn, mx in ops_list:
                val = mn * (1 - self.m/30) + self.m/30 * mx
                xb[b:b+1] = op(xb[b:b+1], val)

        return xb

    def update_prg_trn(self, param_dict, h, indx=None):

        if 'm' in param_dict:
            self.m = (1 - h) * param_dict['m'][0] + h * param_dict['m'][1]

        if 'n' in param_dict:
            self.n = (1 - h) * param_dict['n'][0] + h * param_dict['n'][1]
            self.n = int(self.n + 0.5)


# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.close()
    im = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04\\pod_half\\images\\case_000.npy')
    lb = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04\\pod_half\\labels\\case_000.npy')
    volume = np.stack([im, lb]).astype(np.float32)
    xb = torch.from_numpy(volume[np.newaxis, :, 37:69, 64:192, 64:192]).cuda()
    img = xb[:1]
    aug = torch_myRandAugment(n=1, m=5)
    # %%
    vmin, vmax = img[0, 0].min(), img[0, 0].max()
    img_aug = aug(torch.clone(img))
    plt.subplot(1, 3, 1)
    plt.imshow(img[0, 0, -1].cpu().numpy(), cmap='gray')
    plt.contour(img[0, 1, -1].cpu().numpy(), colors='red')
    plt.subplot(1, 3, 2)
    plt.imshow(img_aug[0, 0, -1].cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.contour(img_aug[0, 1, -1].cpu().numpy(), colors='red')
    plt.subplot(1, 3, 3)
    plt.imshow((img[0, 0, -1] - img_aug[0, 0, -1]).cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.contour(img_aug[0, 1, -1].cpu().numpy(), colors='red')

    # %%
    m = 5
    vmin, vmax = img[0, 0].min(), img[0, 0].max()
    op, mn, mx = aug.all_ops[9]
    print(op)
    val = mn * (1 - m/30) + m/30 * mx
    img_aug = op(torch.clone(img), val)
    plt.subplot(1, 3, 1)
    plt.imshow(img[0, 0, -1].cpu().numpy(), cmap='gray')
    plt.contour(img[0, 1, -1].cpu().numpy(), colors='red')
    plt.subplot(1, 3, 2)
    plt.imshow(img_aug[0, 0, -1].cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.contour(img_aug[0, 1, -1].cpu().numpy(), colors='red')
    plt.subplot(1, 3, 3)
    plt.imshow((img[0, 0, -1] - img_aug[0, 0, -1]).cpu().numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.contour(img_aug[0, 1, -1].cpu().numpy(), colors='red')
