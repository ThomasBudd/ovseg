import numpy as np
import matplotlib.pyplot as plt
from os import listdir, environ
from os.path import join

p = join(environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_2d')
fbp_fols = ['fbps_full', 'fbps_half', 'fbps_quater', 'fbps_eights', 'fbps_16', 'fbps_32']
im_fol = 'images_restauration'
doses = ['full', 'half', 'quater', 'eights', '1/16', '1/32']
# %%
case = 'case_090.npy'
im = np.load(join(p, im_fol, case))
fbps = [np.load(join(p, fbp_fol, case)) for fbp_fol in fbp_fols]
z = 40
plt.subplot(2, 4, 1)
plt.imshow(im[z].astype(float), cmap='bone')
plt.title('vendor')
plt.axis('off')
for i, fbp in enumerate(fbps):
    mse = np.mean((im - fbp)**2)
    psnr = 10 * np.log10(im.ptp()**2/mse)
    if i <= 2:
        plt.subplot(2, 4, 2+i)
    else:
        plt.subplot(2, 4, 3+i)
    plt.imshow(fbp[z].astype(float), cmap='bone')
    plt.title(doses[i] + ' PSNR:{:.2f}'.format(psnr))
    plt.axis('off')