import numpy as np
import torch
from ovseg.utils.torch_np_utils import check_type, stack
from torch.nn.functional import interpolate
from skimage.transform import resize, rescale
from scipy.ndimage import gaussian_filter


class GrayValueAugmentation(object):
    '''
    GrayValueAugmentation(p_noise=0.15, var_noise_mm=[0, 0.1], p_blur=0.15,
                          sigma_blur_mm=[0.5, 1.5], blur_3d=False,
                          p_bright=0.15, fac_bright_mm=[0.7, 1.3],
                          p_contr=0.15, fac_contr_mm=[0.65, 1.5],
                          p_gamma=0.15, gamma_mm=[0.7, 1.5], p_gamma_inv=0.15)
    Performs the following augmentations:
        - adding Gaussian noise
        - blurring with a Gaussian kernel
        - changing the brightness of the image
        - changing the contrast of teh image
        - gamma transformations

    Parameter:
    ----------------
    p_xxx :
        - probability with which xxx is applied to the image
    xxx_mm :
        - min and max of the uniform distribution which is used to draw the
          parameters for xxx
    blur_3d :
        - if True the blurring is applied along all axes, otherwithes only
          in xy plane
    '''

    def __init__(self, p_noise=0.15, var_noise_mm=[0, 0.1],
                 p_blur=0.1, sigma_blur_mm=[0.5, 1.5], blur_3d=False,
                 p_bright=0.15, fac_bright_mm=[0.7, 1.3],
                 p_contr=0.15, fac_contr_mm=[0.65, 1.5],
                 p_gamma=0.15, gamma_mm=[0.7, 1.5], p_gamma_inv=0.15,
                 p_alter_mean_std=0, std_mean=0.1, std_std=0.125,
                 aug_channels=[0]):
        # gaussian noise
        self.p_noise = p_noise
        self.var_noise_mm = var_noise_mm
        # gaussian blur
        self.p_blur = p_blur
        self.sigma_blur_mm = sigma_blur_mm
        self.blur_3d = blur_3d
        # birghtness
        self.p_bright = p_bright
        self.fac_bright_mm = fac_bright_mm
        # contrast
        self.p_contr = p_contr
        self.fac_contr_mm = fac_contr_mm
        # gamma transformation
        self.p_gamma = p_gamma
        self.gamma_mm = gamma_mm
        self.p_gamma_inv = p_gamma_inv
        # altering mean and std
        self.p_alter_mean_std = p_alter_mean_std
        self.std_mean = std_mean
        self.std_std = std_std
        # all channels in this list will be augmented
        self.aug_channels = aug_channels

        # torch filters for gaussian blur
        if self.blur_3d:
            self.gfilter = torch.nn.Conv3d(1, 1, kernel_size=11, bias=False, padding=5)
        else:
            self.gfilter = torch.nn.Conv2d(1, 1, kernel_size=11, bias=False, padding=5)

    def _torch_uniform(self, mm, device='cpu'):
        return (mm[1] - mm[0]) * torch.rand([], device=device) + mm[0]

    def _noise(self, img, is_np):
        if is_np:
            var = np.random.uniform(*self.var_noise_mm)
            sigma = np.sqrt(var)
            return img + sigma*np.random.randn(*img.shape)
        else:
            var = self._torch_uniform(self.var_noise_mm, device=img.device)
            sigma = torch.sqrt(var)
            return img + sigma * torch.randn_like(img)

    def _np_blur_all_axes(self, img):
        sigma = np.random.uniform(*self.sigma_blur_mm)
        return gaussian_filter(img, sigma, mode='constant', cval=img.min())

    def _np_blur_2p5d(self, img):
        sigma = np.random.uniform(*self.sigma_blur_mm)
        return np.stack([gaussian_filter(img[..., z], sigma, mode='constant',
                                         cval=img.min())
                         for z in range(img.shape[2])], -1)

    def _np_blur(self, img):
        if len(img.shape) == 3 and not self.blur_3d:
            return self._np_blur_2p5d(img)
        else:
            return self._np_blur_all_axes(img)

    def _torch_blur_2d(self, img):
        sigma = self._torch_uniform(self.sigma_blur_mm, device=img.device)
        var = sigma ** 2
        axes = torch.arange(-5, 6, device=img.device)
        grid = torch.stack(torch.meshgrid([axes for _ in range(2)]))
        gkernel = torch.exp(-1*torch.sum(grid**2, dim=0)/2.0/var)
        gkernel = gkernel/gkernel.sum()
        gkernel = gkernel.view(1, 1, 11, 11)
        self.gfilter.weight.data = gkernel
        self.gfilter.weight.requires_grad = False
        if len(img.shape) == 2:
            return self.gfilter(img.view(*(1, 1, *img.shape)))[0, 0]
        else:
            img = img.unsqueeze(0).permute(3, 0, 1, 2)
            return self.gfilter(img).permute(1, 2, 3, 0)[0]

    def _torch_blur_3d(self, img):
        sigma = self._torch_uniform(self.sigma_blur_mm, device=img.device)
        var = sigma ** 2
        axes = torch.arange(-5, 6, device=img.device)
        grid = torch.stack(torch.meshgrid([axes for _ in range(3)]))
        gkernel = torch.exp(-1*torch.sum(grid**2, dim=0)/2.0/var)
        gkernel = gkernel/gkernel.sum()
        gkernel = gkernel.view(1, 1, 11, 11, 11)
        self.gfilter.weight.data = gkernel
        self.gfilter.weight.requires_grad = False
        return self.gfilter(img.view(*(1, 1, *img.shape)))[0, 0]

    def _torch_blur(self, img):
        img = img.type(torch.float32)
        if self.blur_3d:
            return self._torch_blur_3d(img)
        else:
            return self._torch_blur_2d(img)

    def _blur(self, img, is_np):
        if is_np:
            return self._np_blur(img)
        else:
            return self._torch_blur(img)

    def _brightness(self, img, is_np):
        if is_np:
            fac = np.random.uniform(*self.fac_bright_mm)
        else:
            fac = self._torch_uniform(self.fac_bright_mm, device=img.device)
        return img * fac

    def _contrast(self, img, is_np):
        if is_np:
            fac = np.random.uniform(*self.fac_contr_mm)
        else:
            fac = self._torch_uniform(self.fac_contr_mm, device=img.device)
        mean = img.mean()
        mn = img.min().item()
        mx = img.max().item()
        img = (img - mean) * fac + mean
        if is_np:
            return img.clip(mn, mx)
        else:
            return img.clamp(mn, mx)

    def _gamma(self, img, invert, is_np):
        if is_np:
            gamma = np.random.uniform(*self.gamma_mm)
        else:
            gamma = self._torch_uniform(self.gamma_mm, device=img.device)
        mn = img.min()
        rng = img.max() - mn
        img = (img - mn)/rng
        if invert:
            img = 1 - img
        if is_np:
            img = img ** gamma
        else:
            img = torch.pow(img, gamma)
        if invert:
            img = 1 - img
        return img * rng + mn

    def _alter_mean_std(self, img, is_np):
        if is_np:
            mean_new = np.random.randn() * self.std_mean
            std_new = np.random.randn() * self.std_std + 1
        else:
            mean_new = torch.randn() * self.std_mean
            std_new = torch.randn() * self.std_std + 1
        mean_old = img.mean()
        std_old = img.std()
        if std_old < 0.625:
            std_old = 0.625
        return (std_new/std_old) * img - mean_old/std_old + mean_new

    def augment_image(self, img):
        '''
        augment_img(img)
        performs grayvalue augmentation for the input image of shape
        (nx, ny(, nz))
        '''
        is_np, _ = check_type(img)
        # first collect what we want to do
        self.do_noise = np.random.rand() < self.p_noise
        self.do_blur = np.random.rand() < self.p_blur
        self.do_bright = np.random.rand() < self.p_bright
        self.do_contr = np.random.rand() < self.p_contr
        self.do_gamma = np.random.rand() < self.p_gamma
        self.do_alter = np.random.rand() < self.p_alter_mean_std
        if self.do_gamma:
            self.invert = np.random.rand() < self.p_gamma_inv
        # Let's-a go!
        if self.do_noise:
            img = self._noise(img, is_np)
        if self.do_blur:
            img = self._blur(img, is_np)
        if self.do_bright:
            img = self._brightness(img, is_np)
        if self.do_contr:
            img = self._contrast(img, is_np)
        if self.do_gamma:
            img = self._gamma(img, self.invert, is_np)
        if self.do_alter:
            img = self._alter_mean_std(img, is_np)
        return img

    def augment_sample(self, sample):
        '''
        augment_sample(sample)
        augments only the first image of the sample as we assume single channel
        images like CT
        '''
        for c in self.aug_channels:
            sample[c] = self.augment_image(sample[c])
        return sample

    def augment_batch(self, batch):
        '''
        augment_batch(batch)
        augments every sample of the batch, in each sample only the image in
        the first channel will be augmented as we assume single channel images
        like CT
        '''
        return stack([self.augment_sample(batch[i])
                      for i in range(len(batch))])

    def augment_volume(self, volume, is_inverse: bool = False):
        if not is_inverse:
            if len(volume.shape) == 3:
                volume = self.augment_image(volume)
            else:
                volume = self.augment_sample(volume)
        return volume


# %%
class torch_gray_value_augmentation(torch.nn.Module):

    def __init__(self,
                 p_noise=0.15,
                 p_blur=0.1,
                 p_bright=0.15,
                 p_contr=0.15,
                 p_low_res=0.125,
                 p_gamma=0.15,
                 p_gamma_invert=0.15,
                 mm_var_noise=[0, 0.1],
                 mm_sigma_blur=[0.5, 1.5],
                 mm_bright=[0.7, 1.3],
                 mm_contr=[0.65, 1.5],
                 mm_low_res=[1, 2],
                 mm_gamma=[0.7, 1.5],
                 n_im_channels: int = 1
                 ):
        super().__init__()
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_bright = p_bright
        self.p_contr = p_contr
        self.p_low_res = p_low_res
        self.p_gamma = p_gamma
        self.p_gamma_invert = p_gamma_invert
        self.mm_var_noise = mm_var_noise
        self.mm_sigma_blur = mm_sigma_blur
        self.mm_bright = mm_bright
        self.mm_contr = mm_contr
        self.mm_low_res = mm_low_res
        self.mm_gamma = mm_gamma
        self.n_im_channels = n_im_channels

    def _uniform(self, mm, device='cpu'):
        return (mm[1] - mm[0]) * torch.rand([], device=device) + mm[0]

    def _noise(self, img):
        var = self._uniform(self.mm_var_noise, device=img.device)
        sigma = torch.sqrt(var)
        return img + sigma * torch.randn_like(img)

    def _blur(self, img):
        sigma = self._uniform(self.mm_sigma_blur, device=img.device)
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

    def _brightness(self, img):
        fac = self._uniform(self.mm_bright, device=img.device)
        return img * fac

    def _contrast(self, img):
        fac = self._uniform(self.mm_contr, device=img.device)
        mean = img.mean()
        mn = img.min().item()
        mx = img.max().item()
        img = (img - mean) * fac + mean
        return img.clamp(mn, mx)

    def _low_res(self, img):
        size = img.size()[2:]
        mode = 'bilinear' if len(size) == 2 else 'trilinear'
        fac = np.random.uniform(*self.mm_low_res)
        img = interpolate(img, scale_factor=1/fac)
        return interpolate(img, size=size, mode=mode)

    def _gamma(self, img):
        with torch.cuda.amp.autocast(enabled=False):
            mn, mx = img.min(), img.max()
            img = (img - mn)/(mx - mn)
            gamma = np.random.uniform(*self.mm_gamma)
            if np.random.rand() < self.p_gamma_invert:
                img = 1 - (1 - img) ** gamma
            else:
                img = img ** gamma

            return (mx - mn) * img + mn

    def _get_ops_list(self):
        ops_list = []
        if np.random.rand() < self.p_noise:
            ops_list.append(self._noise)
        if np.random.rand() < self.p_blur:
            ops_list.append(self._blur)
        if np.random.rand() < self.p_bright:
            ops_list.append(self._brightness)
        if np.random.rand() < self.p_contr:
            ops_list.append(self._contrast)
        if np.random.rand() < self.p_low_res:
            ops_list.append(self._low_res)
        np.random.shuffle(ops_list)

        return ops_list

    def forward(self, xb):

        c = self.n_im_channels

        for b in range(xb.shape[0]):
            for op in self._get_ops_list():
                xb[b:b+1, :c] = op(xb[b:b+1, :c])
        return xb

    def update_prg_trn(self, param_dict, h, indx=None):

        attr_list = ['p_noise', 'p_blur', 'p_bright', 'p_contr', 'p_low_res', 'mm_var_noise',
                     'mm_sigma_blur', 'mm_bright', 'mm_contr', 'mm_low_res']

        for attr in attr_list:
            if attr in param_dict:
                self.__setattr__(attr, (1 - h) * param_dict[attr][0] + h * param_dict[attr][1])


# %%
class np_gray_value_augmentation():

    def __init__(self,
                 p_noise=0.15,
                 p_blur=0.1,
                 p_bright=0.15,
                 p_contr=0.15,
                 p_low_res=0.125,
                 p_gamma=0.15,
                 p_gamma_invert=0.15,
                 mm_var_noise=[0, 0.1],
                 mm_sigma_blur=[0.5, 1.5],
                 mm_bright=[0.7, 1.3],
                 mm_contr=[0.65, 1.5],
                 mm_low_res=[1, 2],
                 mm_gamma=[0.7, 1.5],
                 n_im_channels: int = 1
                 ):
        super().__init__()
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_bright = p_bright
        self.p_contr = p_contr
        self.p_low_res = p_low_res
        self.p_gamma = p_gamma
        self.p_gamma_invert = p_gamma_invert
        self.mm_var_noise = mm_var_noise
        self.mm_sigma_blur = mm_sigma_blur
        self.mm_bright = mm_bright
        self.mm_contr = mm_contr
        self.mm_low_res = mm_low_res
        self.mm_gamma = mm_gamma
        self.n_im_channels = n_im_channels

    def _noise(self, img):
        sigma = np.sqrt(np.random.uniform(*self.mm_var_noise))
        return img + sigma * np.random.randn(*img.shape)

    def _blur(self, img):

        if len(img.shape) == 5:
            # 3d images
            for b in range(img.shape[0]):
                for c in range(img.shape[1]):
                    sigma = np.random.uniform(*self.mm_sigma_blur)
                    for z in range(img.shape[2]):
                        img[b, c, z] = gaussian_filter(img[b, c, z], sigma, mode='constant',
                                                       cval=img[b, c, z].min())
        else:
            # 2d images
            for b in range(img.shape[0]):
                for c in range(img.shape[1]):
                    sigma = np.random.uniform(*self.mm_sigma_blur)
                    img[b, c] = gaussian_filter(img[b, c], sigma, mode='constant',
                                                cval=img[b, c].min())
        return img

    def _brightness(self, img):
        fac = np.random.uniform(*self.mm_bright)
        return img * fac

    def _contrast(self, img):
        fac = np.random.uniform(*self.mm_contr)
        mean = img.mean()
        mn = img.min().item()
        mx = img.max().item()
        img = (img - mean) * fac + mean
        return img.clip(mn, mx)

    def _low_res(self, img):
        orig_shape = img.shape[2:]

        if len(img.shape) == 5:
            # 3d images
            for b in range(img.shape[0]):
                for c in range(img.shape[1]):
                    scale = 1 / np.random.uniform(*self.mm_low_res)
                    for z in range(img.shape[2]):
                        img_low = rescale(img[b, c, z], scale=scale, order=0)
                        img[b, c, z] = resize(img_low, orig_shape, order=3)
        else:
            # 2d images
            for b in range(img.shape[0]):
                for c in range(img.shape[1]):
                    scale = 1 / np.random.uniform(*self.mm_low_res)
                    img_low = rescale(img[b, c], scale=scale, order=0)
                    img[b, c] = resize(img_low, orig_shape, order=3)
        return img

    def _gamma(self, img):
        mn, mx = img.min(), img.max()
        img = (img - mn)/(mx - mn)
        gamma = np.random.uniform(*self.mm_gamma)
        if np.random.rand() < self.p_gamma_invert:
            img = 1 - (1 - img) ** gamma
        else:
            img = img ** gamma

        return (mx - mn) * img + mn

    def _get_ops_list(self):
        ops_list = []
        if np.random.rand() < self.p_noise:
            ops_list.append(self._noise)
        if np.random.rand() < self.p_blur:
            ops_list.append(self._blur)
        if np.random.rand() < self.p_bright:
            ops_list.append(self._brightness)
        if np.random.rand() < self.p_contr:
            ops_list.append(self._contrast)
        if np.random.rand() < self.p_low_res:
            ops_list.append(self._low_res)
        if np.random.rand() < self.p_gamma:
            ops_list.append(self._gamma)
        np.random.shuffle(ops_list)

        return ops_list

    def __call__(self, xb):

        c = self.n_im_channels

        for b in range(xb.shape[0]):
            for op in self._get_ops_list():
                xb[b:b+1, :c] = op(xb[b:b+1, :c])
        return xb

    def update_prg_trn(self, param_dict, h, indx=None):

        attr_list = ['p_noise', 'p_blur', 'p_bright', 'p_contr', 'p_low_res', 'p_gamma',
                     'p_gamma_invert', 'mm_var_noise', 'mm_sigma_blur', 'mm_bright', 'mm_contr',
                     'mm_low_res', 'mm_gamma']

        for attr in attr_list:
            if attr in param_dict:
                self.__setattr__(attr, (1 - h) * param_dict[attr][0] + h * param_dict[attr][1])


# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
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
    aug = torch_gray_value_augmentation(p_noise=0.5, p_blur=0.5, p_bright=0.5, p_contr=0.5)

    # %%
    xb_aug = aug(xb).cpu().numpy()

    z = np.argmax(np.sum(lb_crop > 0, (1, 2)))
    plt.subplot(1, 2, 1)
    plt.imshow(im_crop[z], cmap='gray')
    plt.contour(lb_crop[z])
    plt.subplot(1, 2, 2)
    plt.imshow(xb_aug[0, 0, z], cmap='gray')
    plt.contour(xb_aug[0, 1, z])
