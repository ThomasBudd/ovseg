from os import environ, listdir
from os.path import join
from skimage.measure import label
import numpy as np
import nibabel as nib
from tqdm import tqdm

lbp = join(environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'labels')
cl = 9

vols9 = []
vols1 = []

for case in tqdm(listdir(lbp)):
    img = nib.load(join(lbp, case))
    fac = np.prod(img.header['pixdim'][1:4]) / 1000
    lb9 = (img.get_fdata() == 9).astype(float)
    lb1 = (img.get_fdata() == 1).astype(float)
    comps9 = label(lb9)
    comps1 = label(lb1)
    for c in range(1, comps9.max() + 1):
        vols9.append(np.sum(comps9 == c) * fac)
    for c in range(1, comps1.max() + 1):
        vols1.append(np.sum(comps1 == c) * fac)

vols9_sort = np.sort(vols9)[::-1]
vols1_sort = np.sort(vols1)[::-1]
# %%

print('9: mean: {:.1f}, med: {:.1f}'.format(np.mean(vols9_sort), np.median(vols9_sort)))
print('1: mean: {:.1f}, med: {:.1f}'.format(np.mean(vols1_sort), np.median(vols1_sort)))

# %%
vols_sort = vols9_sort
total_vol = np.sum(vols_sort)
cum_vol = np.cumsum(vols_sort)
p=0.95
n = 150
k_p = np.sum(cum_vol < total_vol * p)
p_vol = vols_sort[k_p]
print('{:.1f}cm3 includes {} comps'.format(p_vol, k_p))
print('{}th biggest: {:.1f}cm3'.format(n, vols_sort[n-1]))