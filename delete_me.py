import os
import numpy as  np
import matplotlib.pyplot as plt

pp = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_half')

im_folders = [f for f in os.listdir(pp) if f.startswith('image')]
lb_folders = [f for f in os.listdir(pp) if f.startswith('label')]

# %%
scan = np.random.choice(os.listdir(os.path.join(pp, 'images')))

ims = [np.load(os.path.join(pp, imf, scan)) for imf in im_folders]
lbs = [np.load(os.path.join(pp, lbf, scan)) for lbf in lb_folders]

for i, (im, lb, fol) in enumerate(zip(ims, lbs, im_folders)):
    plt.subplot(1, 3, 1+i)
    if len(im.shape) == 4:
        im = im[0]
        lb = lb[0]
    z = np.argmax(np.sum(lb, (1, 2)))    
    plt.imshow(im[z].astype(float), cmap='gray')
    plt.contour(lb[z] > 0)
    plt.title(fol)
