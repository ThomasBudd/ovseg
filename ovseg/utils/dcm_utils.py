from os import listdir
from os.path import join
import pydicom

def is_im_dcm_ds(ds):
    attrs = ['pixel_array', 'ImagePositionPatient', 'PixelSpacing',
             'RescaleSlope', 'RescaleIntercept']
    for attr in attrs:
        if not hasattr(ds, attr):
            return False
    return True


def is_roi_dcm_ds(ds):
    attrs = ['StructureSetROISequence', 'ROIContourSequence']
    for attr in attrs:
        if not hasattr(ds, attr):
            return False
    return True


def read_and_split_dcms(dcm_folder):

    series_data = []
    used_files = []
    for file in listdir(dcm_folder):
        try:
            series_data.append(pydicom.dcmread(join(dcm_folder, file)))
            used_files.append(join(dcm_folder, file))
        except Exception:
            # not a valid dcm file
            continue
    
    im_series, roi_series, unidentified = [], [], []

    for ds, file in zip(series_data, used_files):
        if is_im_dcm_ds(ds):
            im_series.append(ds)
        elif is_roi_dcm_ds(ds):
            roi_series.append(ds)
        else:
            unidentified.append(file)

    if len(unidentified) > 0:
        raise TypeError('Something went wrong while reading dcm data from {}. Could not identify '
                        'the following dcm files as image or ROI dcms: {}.'.format(dcm_folder,
                                                                                   unidentified))
        
    return im_series, roi_series

# %%
from os import listdir, environ
from os.path import join

rp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS_dcm')

dcmp = join(rp, listdir(rp)[7])

im_series, roi_series = read_and_split_dcms(dcmp)
rtds = roi_series[0]
ds = im_series[0]
print(ds.PatientPosition)

for ds in im_series:
    print(ds.SliceLocation, ds.ImagePositionPatient[2])
# %%
for scan in listdir(rp):
    dcmp = join(rp, scan)
    ds = pydicom.dcmread(join(dcmp, listdir(dcmp)[2]))
    print(scan, ds.PatientPosition, ds.Manufacturer)
