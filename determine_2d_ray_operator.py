import torch
from torch_radon import RadonFanbeam
import numpy as np
import matplotlib.pyplot as plt
import os

im = np.load(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04',
                          'pod_full', 'images', 'case_000.npy'))

im = im[:, np.newaxis]

# %% detmine n_angles

n_angles = 256
det_count = 736
source_distance = 595
det_distance = 1085.6-595
det_spacing = 1

radon = RadonFanbeam(resolution=512,
                     angles=np.linspace(0, np.pi, n_angles, endpoint=False),
                     source_distance=source_distance,
                     det_distance=det_distance,
                     det_count=det_count,
                     det_spacing=det_spacing)


def fbp(y):
    return radon.backprojection(radon.filter_sinogram(y))
y = radon.forward(torch.from_numpy(im).cuda().type(torch.float))

im_fbp = fbp(y).cpu().numpy()

for i, z in enumerate(np.random.choice(list(range(im.shape[0])), size=3)):
    plt.subplot(2, 3, i+1)
    plt.imshow(im[z, 0].clip(-150, 250), cmap='gray')
    plt.subplot(2, 3, i+4)
    plt.imshow(im_fbp[z, 0].clip(-150, 250), cmap='gray')
    
# %%