import numpy as np
from skimage.measure import label
import nibabel as nib
from os import listdir, environ
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt

lbp = join(environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'labels')

bbox_pod, bbox_om = [], []
vol_pod, vol_om = [], []

for case in tqdm(listdir(lbp)):
    img = nib.load(join(lbp, case))
    seg = img.get_fdata()
    pod = (seg == 9)
    om = (seg == 1)
    sp = img.header['pixdim'][1:4]
    vf = np.prod(sp)
    
    pod_comps = label(pod)
    for c in range(1, pod_comps.max()+1):
        comp = (pod_comps == c)
        vol_pod.append(np.sum(comp) * vf)
        bbox_pod.append(np.stack(np.where(comp), 1).ptp(0) * sp)

    om_comps = label(om)
    for c in range(1, om_comps.max()+1):
        comp = (om_comps == c)
        vol_om.append(np.sum(comp) * vf)
        bbox_om.append(np.stack(np.where(comp), 1).ptp(0) * sp)

bbox_pod = np.array(bbox_pod)
bbox_om = np.array(bbox_om)
vol_pod = np.array(vol_pod)
vol_om = np.array(vol_om)

# %%
t1, t2 = 1000, 5000
plt.close()
plt.subplot(2, 2, 1)
plt.hist(vol_pod[vol_pod > t1])
plt.title('pod')
plt.subplot(2, 2, 2)
plt.hist(vol_om[vol_om > t1])
plt.title('om')
plt.subplot(2, 2, 3)
plt.hist(vol_pod[vol_pod > t2])
plt.subplot(2, 2, 4)
plt.hist(vol_om[vol_om > t2])

print('POD cc: {:.3f}% > {} mm^2, {:.3f}% > {} mm^2'.format(100 * np.mean(vol_pod > t1), t1,
                                                            100 * np.mean(vol_pod > t2), t2))
print('POD vol: {:.3f}% > {} mm^2, {:.3f}% > {} mm^2'.format(100 * np.sum(vol_pod[vol_pod > t1])/np.sum(vol_pod), t1,
                                                             100 * np.sum(vol_pod[vol_pod > t2])/np.sum(vol_pod), t2))
print('omn cc: {:.3f}% > {} mm^2, {:.3f}% > {} mm^2'.format(100 * np.mean(vol_om > t1), t1,
                                                            100 * np.mean(vol_om > t2), t2))
print('omn vol: {:.3f}% > {} mm^2, {:.3f}% > {} mm^2'.format(100 * np.sum(vol_om[vol_om > t1])/np.sum(vol_om), t1,
                                                             100 * np.sum(vol_om[vol_om > t2])/np.sum(vol_om), t2))
    
# %%
plt.figure()
for i in range(3):
    plt.subplot(2, 3, i + 1)
    plt.hist(bbox_pod[:, i])
    plt.subplot(2, 3, i + 4)
    plt.hist(bbox_om[:, i])