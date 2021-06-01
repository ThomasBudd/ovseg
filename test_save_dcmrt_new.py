from ovseg.utils.io import read_dcms
import pydicom
import numpy as np

dcmp = 'D:\\PhD\\Data\\Apollo_Hilal\\AP-P5L4'
data_tpl = read_dcms(dcmp)
dcmrtp = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\dcm_rt_test\\dcms\\rtstruct_test.dcm'

# %%
from rt_utils import RTStructBuilder

# Create new RT Struct. Requires the DICOM series path for the RT Struct.
rtstruct = RTStructBuilder.create_new(dicom_series_path=dcmp)

# ...
# Create mask through means such as ML
# ...

# Add the 3D mask as an ROI.
# The colour, description, and name will be auto generated
rtstruct.add_roi(mask=np.moveaxis(data_tpl['label'], 0, -1) > 0)

rtstruct.save('new_rt_struct')


# %%
ds = pydicom.dcmread('new_rt_struct.dcm')