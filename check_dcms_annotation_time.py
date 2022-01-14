import os
import pydicom
import numpy as np
# from ovseg.utils.io import read_dcms

dp = 'D:\PhD\Data\ov_data_base\\raw_data\\ICON8_14_Derby_Burton'

scans = os.listdir(dp)

spacings = []

weird_scans = []
for scan in scans:
    
    dcms = os.listdir(os.path.join(dp, scan))
    ds_list = [pydicom.dcmread(os.path.join(dp, scan, dcm)) for dcm in dcms]
    z_im = [ds.ImagePositionPatient[2] for ds in ds_list]
    diff = np.diff(np.sort(z_im))
    print(scan)
    print('{:.3f}, {:.3f}, {:.3f}'.format(np.min(diff), np.median(diff), np.max(diff)))
    if np.min(diff) < np.median(diff):
        weird_scans.append(scan)
    spacings.append(np.median(diff))

spacings = np.round(np.array(spacings),1)

# %%
scan = weird_scans[0]
dcms = os.listdir(os.path.join(dp, scan))
ds_list = [pydicom.dcmread(os.path.join(dp, scan, dcm)) for dcm in dcms]
z_im = [ds.ImagePositionPatient[2] for ds in ds_list]
diff = np.diff(np.sort(z_im))
