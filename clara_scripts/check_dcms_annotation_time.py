import os
import pydicom
import numpy as np
# from ovseg.utils.io import read_dcms
from tqdm import tqdm

dp = 'D:\PhD\Data\ov_data_base\\raw_data\\ICON8_14_Derby_Burton'

#scans = os.listdir(dp)
scans = []
for root, dirs, files in os.walk(dp):
    if len(files) > 5:
        scans.append(root)

spacings = []
slice_thicknesses = []
weird_scans = []
thin_slices = []

for scan in tqdm(scans):
    
    dcms = [dcm for dcm in os.listdir(os.path.join(dp, scan)) if dcm.endswith('.dcm')]
    ds_list = [pydicom.dcmread(os.path.join(dp, scan, dcm)) for dcm in dcms]
    z_im = [ds.ImagePositionPatient[2] for ds in ds_list]
    diff = np.diff(np.sort(z_im))
    print(scan)
    print('{:.3f}, {:.3f}, {:.3f}'.format(np.min(diff), np.median(diff), np.max(diff)))
    if np.min(diff) < np.median(diff):
        weird_scans.append(scan)
    
    if np.median(diff) < 5.0:
        thin_slices.append(scan)
    spacings.append(np.median(diff))
    slice_thicknesses.append(ds_list[0].SliceThickness)

spacings = np.round(np.array(spacings),1)

# %%
scan = weird_scans[0]
dcms = os.listdir(os.path.join(dp, scan))
ds_list = [pydicom.dcmread(os.path.join(dp, scan, dcm)) for dcm in dcms]
z_im = [ds.ImagePositionPatient[2] for ds in ds_list]
diff = np.round(np.diff(np.sort(z_im)),2)
