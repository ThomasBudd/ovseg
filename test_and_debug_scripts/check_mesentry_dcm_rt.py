import numpy as np
import os
from tqdm import tqdm
import pydicom
from ovseg.utils.io import _is_roi_dcm_ds
from ovseg.data.Dataset import raw_Dataset
import matplotlib.pyplot as plt

les_names = []
dp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'mesentery')
for scan in tqdm(os.listdir(dp)):
    if scan.endswith('.txt'):
        continue
    scanp = os.path.join(dp, scan)
    dcms = [os.path.join(scanp, dcm) for dcm in os.listdir(scanp) if dcm.endswith('dcm')]
    dcms = [dcm for dcm in dcms if os.path.basename(dcm).startswith('ID_')]
    ds = pydicom.dcmread(dcms[0])
    if hasattr(ds, 'StructureSetROISequence'):
        names_found = [s.ROIName.lower() for s in ds.StructureSetROISequence]
        print(names_found)
        les_names.extend(names_found)
        les_names = np.unique(les_names).tolist()

for name in sorted(les_names):
    print(name)

# %%
ds = raw_Dataset(dp, dcm_names_dict={'100': 1})

# %%
data_tpl = ds[5]
lb = data_tpl['label']
im = data_tpl['image'].clip(-150, 250)
x = np.argmax(np.sum(lb, (0, 2)))
y = np.argmax(np.sum(lb, (0, 1)))
z = np.argmax(np.sum(lb, (1, 2)))
plt.subplot(1, 3, 1)
plt.imshow(im[z], cmap='bone')
plt.contour(lb[z])
plt.subplot(1, 3, 2)
plt.imshow(im[:, x], cmap='bone')
plt.contour(lb[:, x])
plt.subplot(1, 3, 3)
plt.imshow(im[...,y], cmap='bone')
plt.contour(lb[...,y])