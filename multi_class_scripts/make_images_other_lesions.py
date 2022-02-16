from os import listdir, environ
from os.path import join
import nibabel as nib
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

lbp = join(environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'labels')

classes = [2, 3, 5, 13, 15, 17]
all_vols = []

for case in tqdm(listdir(lbp)):
    
    img = nib.load(join(lbp, case))
    fac = np.prod(img.header['pixdim'][1:4])
    gt = img.get_fdata()
    
    vols = np.zeros(len(classes))
    for i, cl in enumerate(classes):
        lb = (gt == cl).astype(float)
        vol = np.sum(lb) * fac
        
        if vol > 0:
            vols[i] = vol
        else:
            vols[i] = np.nan
    all_vols.append(vols)

all_vols = np.array(all_vols)

# %%

med_vol = np.nanmean(all_vols, 0)

k_med_vol = np.nanargmin(np.abs(all_vols - med_vol.reshape(1, -1)), 0)

data_info = pickle.load(open(join(environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'data_info.pkl'), 'rb'))

for i in range(len(classes)):
    
    k = k_med_vol[i]
    cl = classes[i]
    info = data_info['case_%03d'%k]
    print('class {}: id: {}, date: {}, vol: {:.1f}'.format(cl,
                                                       info['pat_id'],
                                                       info['date'],
                                                       med_vol[i]/1000))
    
# %%
ims = []
lbs = []
for i in range(6):
    k = k_med_vol[i]
    case = 'case_%03d'%k
    im = nib.load(join(environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'images', case+'_0000.nii.gz')).get_fdata()
    lb = nib.load(join(environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'labels', case+'.nii.gz')).get_fdata()
    ims.append(im)
    lbs.append(lb)
# %%

i = 0
y_mm = [0, 512]
z_mm = [0, 99]
im = ims[0].clip(-50, 150)
lb = lbs[0]
cl = classes[i]
x = np.argmax(np.sum(lb == cl, (1, 2)))

plt.imshow(im[x, y_mm[0]:y_mm[1], z_mm[0]:z_mm[1]], cmap='bone')
plt.contour(lb[x, y_mm[0]:y_mm[1], z_mm[0]:z_mm[1]] == i, colors='red')
