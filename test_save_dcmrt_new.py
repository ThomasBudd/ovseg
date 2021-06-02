from ovseg.utils.io import read_dcms
import pydicom
import numpy as np
import os
import nibabel as nib

dcmp = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\BARTS_dcm\\ID_133_1'
data_tpl = read_dcms(dcmp)

pred = nib.load(os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 'OV04', 'pod_half',
                             'res_encoder', 'BARTSensemble_0_1_2_3_4',
                             'case_313.nii.gz')).get_data()
pod = (np.moveaxis(data_tpl['label'], 0, -1) == 9).astype(float)

print(200 * np.sum(pred * pod) / np.sum(pred + pod))

# %%
from rt_utils import RTStructBuilder

rtstruct = RTStructBuilder.create_from(
  dicom_series_path=dcmp, 
  rt_struct_path=os.path.join(dcmp, 'ID_133_1_NEW.dcm')
)

# Add ROI. This is the same as the above example.
rtstruct.add_roi(
  mask=pod>0, 
  color=[255, 0, 255], 
  name="9-POD automated"
)

rtstruct.save('new-rt-struct')

# %%
ds = pydicom.dcmread('pod_contours.dcm')