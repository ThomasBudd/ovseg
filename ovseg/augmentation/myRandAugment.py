import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class torch_myRandAugment(nn.Module):

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
    aug = medRandAugment(n=2, m=10)
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
