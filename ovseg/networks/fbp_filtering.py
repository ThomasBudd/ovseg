import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import scipy.fft

    fftmodule = scipy.fft
except ImportError:
    import numpy.fft

    fftmodule = numpy.fft


# %%
class fbp_filtering(nn.Module):

    def __init__(self, det_size):
        super().__init__()
        self.det_size = det_size


        padded_size = max(64, int(2 ** np.ceil(np.log2(2 * self.det_size))))

        # define ramp filter
        n = np.concatenate((np.arange(1, padded_size / 2 + 1, 2, dtype=np.int),
                    np.arange(padded_size / 2 - 1, 0, -2, dtype=np.int)))
        f = np.zeros(padded_size)
        f[0] = 0.25
        f[1::2] = -1 / (np.pi * n) ** 2

        self.ramp_filter = 2 * np.real(fftmodule.fft(f))
        self.ramp_filter = torch.from_numpy(self.ramp_filter).type(torch.float)
        # now the learned fourier filter
        self.filter = nn.Parameter(torch.zeros((padded_size)))

    def forward(self, sinogram):
        size = sinogram.size(3)
        n_angles = sinogram.size(2)

        padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
        pad = padded_size - size

        padded_sinogram = F.pad(sinogram.float(), (0, pad, 0, 0))
        sino_fft = torch.rfft(padded_sinogram, 2, normalized=True, onesided=False)

        # get filter and apply
        fourier_filter = self.ramp_filter.to(sinogram.device) * (0.5 + self.filter)
        fourier_filter = fourier_filter.view(1, 1, -1, 1).type(torch.float)
        filtered_sino_fft = sino_fft * fourier_filter

        # Inverse fft
        filtered_sinogram = torch.irfft(filtered_sino_fft, 1, normalized=True, onesided=False)

        # pad removal and rescaling
        filtered_sinogram = filtered_sinogram[:, :, :, :-pad] * (np.pi / (2 * n_angles))

        return filtered_sinogram.to(dtype=sinogram.dtype)

