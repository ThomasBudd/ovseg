import torch
import numpy as np
from torch_radon import RadonFanbeam
from os import environ, listdir
from os.path import join
import nibabel as nib
import torch.nn.functional as F
try:
    from tqdm import tqdm
except ImportError:
    print('tqdm not installed. Not pretty progressing bars')
    tqdm = lambda x: x
from ovseg.utils.io import save_pkl, save_txt, read_nii

n_volumes = 25

window = [-32, 318]
mu_water= 0.0192
operator = RadonFanbeam(512,
                        np.linspace(0,2*np.pi, 500),
                        source_distance=600, 
                        det_count=736,
                        det_spacing=1.0)
dose_level = 1.0
bowtie_filt = np.load('default_photon_stats.npy')/64
bowtie_filt = dose_level * torch.from_numpy(bowtie_filt).cuda()

# %%
class sino_filter_base():

    def get_sino_sino_filter():
        raise NotImplementedError()

    def __call__(self, sinogram):
        size = sinogram.size(2)
        n_angles = sinogram.size(1)

        padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
        pad = padded_size - size

        padded_sinogram = F.pad(sinogram.float(), (0, pad, 0, 0))
        sino_fft = torch.rfft(padded_sinogram, 1, normalized=True, onesided=False)

        # get filter and apply
        filtered_sino_fft = sino_fft * self.get_sino_sino_filter()

        # Inverse fft
        filtered_sinogram = torch.irfft(filtered_sino_fft, 1, normalized=True, onesided=False)

        # pad removal and rescaling
        filtered_sinogram = filtered_sinogram[:, :, :-pad] * (np.pi / (2 * n_angles))

        return filtered_sinogram.to(dtype=sinogram.dtype)

# %% DO SOMETHING HERE!
filter_fctns = [operator.filter_sinogram]
filter_fctns_names = ['ramp']


# %%
def simulate_psnr(img, filter_fctn):
    # img in HU
    # function for filtering the sinogram
    img_hu = img.copy()
    img = torch.from_numpy(img).type(torch.float).clip(-1000)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # rescale from HU to linear attenuation
    img_linatt = (img + 1000) / 1000 * mu_water
    img_linatt = img_linatt.type(torch.float).to(dev)

    # rescale from HU to linear attenuation
    proj = operator.forward(img_linatt)
    proj = torch.exp(-proj) 
    proj = torch.poisson(bowtie_filt.expand_as(proj)*proj)*(1/bowtie_filt.expand_as(proj))
    sinogram_noisy = -torch.log(1e-6 + proj)
    fbp_linatt = operator.backprojection(filter_fctn(sinogram_noisy))
    fbp = 1000 * (fbp_linatt - mu_water) / mu_water
    fbp = fbp.cpu().numpy()
    mse = np.mean((img_hu - fbp)**2)
    psnr = 10 * np.log10(img_hu.ptp()**2 / mse)
    img_win = img_hu.clip(*window)
    fbp_win = fbp.clip(*window)
    mse_win = np.mean((img_win - fbp_win)**2)
    psnr_win = 10 * np.log10(img_win.ptp()**2 / mse_win)
    
    return np.array([psnr, psnr_win])

# %%
p = join(environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'images')
cases = listdir(p)[:n_volumes]

results = {name: 0 for name in filter_fctns_names}

for case in tqdm(cases):
    img = read_nii(join(p, case))[0][np.newaxis]

    for filter_fctn, name in zip(filter_fctns, filter_fctns_names):
        results[name] += simulate_psnr(img, filter_fctn)

for name in results:
    results[name] /= len(cases)

save_pkl(results, 'FBP_PSNR_simulations.pkl')
save_txt(results, 'FBP_PSNR_simulations.txt')