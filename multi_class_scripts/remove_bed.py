import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.measure import label
from ovseg.utils.torch_morph import opening_2d
from scipy.ndimage.morphology import binary_fill_holes
import torch
from tqdm import tqdm

rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04')
plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'remove_table')
if not os.path.exists(plotp):
    os.makedirs(plotp)
imp = os.path.join(rawp, 'images')
lbp = os.path.join(rawp, 'labels')

case = 'case_000'
im = nib.load(os.path.join(imp, case+'_0000.nii.gz')).get_fdata()
im = np.moveaxis(im, -1, 0)
lb = nib.load(os.path.join(lbp, case+'.nii.gz')).get_fdata()
lb = np.moveaxis(lb, -1, 0)

mask = im[np.newaxis] > -800
k=6
circ = (np.sum(np.stack(np.meshgrid(np.linspace(-1,1,2*k+1),np.linspace(-1,1,2*k+1)))**2,0) <= 1).astype(float)


mask = opening_2d(torch.from_numpy(im[np.newaxis] > -800).cuda(), circ)[0].cpu().numpy()
mask = np.stack([binary_fill_holes(mask[z]) for z in range(mask.shape[0])])

im = (im + 1000) * mask - 1000

im = im.clip(-31, 318)
# %%
c = 0
mean_lim = -31 * (1-c) + 318 * c

outliers = (im == -31).astype(float) + (im == 318).astype(float)

inside = 1 - outliers

x = np.where(np.mean(inside, (1) ).max(0) > c)[0]
y = np.where(np.mean(inside, (2) ).max(0) > c)[0]
print(np.min(y), np.max(y), np.max(y) - np.min(y))
print(np.min(x), np.max(x), np.max(x) - np.min(x))

# %%
im = im[:, y.min():y.max()+1, x.min():x.max()+1]

for z in range(im.shape[0]):
    plt.imshow(im[z], cmap='bone')
    plt.axis('off')
    plt.savefig(os.path.join(plotp, case+'_'+str(z)))
    plt.close()

# %%
x,y=np.where(np.sum(lb,0) > 0)
print(np.min(y), np.max(y), np.max(y) - np.min(y))
print(np.min(x), np.max(x), np.max(x) - np.min(x))

# %%

def get_dist_to_im_bdry(lb):
    x, y = np.where(np.sum(lb,0) > 0)
    return x.min(), lb.shape[1] - x.max(), y.min(), lb.shape[2] - y.max()

dists = []

for case in tqdm(os.listdir(lbp)):
    lb = nib.load(os.path.join(lbp, case)).get_fdata()
    lb = np.moveaxis(lb, -1, 0)
    if lb.max() > 0:
        dists.append(get_dist_to_im_bdry(lb))

print(np.min(dists, 0))

