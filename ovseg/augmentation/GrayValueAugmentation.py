import numpy as np
import torch
from ovseg.utils.torch_np_utils import check_type, stack
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    print('Caught Import Error while importing some function from scipy or skimage. '
          'Please use a newer version of gcc.')

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
