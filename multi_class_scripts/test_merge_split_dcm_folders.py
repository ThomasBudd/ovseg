import os
import pydicom
import numpy as np

bp = 'D:\\PhD\\Data\\TCGA_new_RW\\TCGA-13-0720'
abdp = os.path.join(bp, 'abdomen')
pelp = os.path.join(bp, 'pelvis')

ds_list1 = [pydicom.dcmread(os.path.join(abdp, dcm)) for dcm in 
            os.listdir(abdp) if not dcm.startswith('TCGA')]
ds_list2 = [pydicom.dcmread(os.path.join(pelp, dcm)) for dcm in 
            os.listdir(pelp) if not dcm.startswith('TCGA')]

# %%
def print_diff_attr(ds_list1, ds_list2):
    for attr in ['AcquisitionDate', 'ConvolutionKernel', 'ImagePositionPatient',
                 'ImageOrientationPatient', 'ManufacturerModelName',
                 'SOPClassUID', 'SOPInstanceUID', 'SeriesInstanceUID',
                 'SliceThickness', 'PixelSpacing', 'StudyID', 'StudyInstanceUID']:
        
        a1, a2 = ds_list1[0].__getattr__(attr), ds_list2[0].__getattr__(attr)
        
        if not a1 == a2:
            
            print(attr)
            print(ds_list1[0].__getattr__(attr))
            print(ds_list2[0].__getattr__(attr))
print_diff_attr(ds_list1, ds_list2)
# %%
for ds in ds_list1:
    print(ds.ImagePositionPatient[2])
    
for ds in ds_list2:
    print(ds.ImagePositionPatient[2])
# %%
st1 = np.median(np.diff([ds.ImagePositionPatient[2] for ds in ds_list1]))
st2 = np.median(np.diff([ds.ImagePositionPatient[2] for ds in ds_list2]))

# %%
print(np.diff([ds.ImagePositionPatient[2] for ds in ds_list1]))
print(np.diff([ds.ImagePositionPatient[2] for ds in ds_list2]))

# %%

st = -7.5
z_start = float(ds_list1[-1].ImagePositionPatient[2])
print(z_start)
for i, ds in enumerate(ds_list2):
    print(z_start + (i+1) * st)
# %%
mp = 'D:\\PhD\\Data\\TCGA_new_RW_merged\\TCGA-13-0720'
st = -7.5
z_start = float(ds_list1[-1].ImagePositionPatient[2])
for i, ds in enumerate(ds_list2):
    ds.ImagePositionPatient[2] = pydicom.valuerep.DSfloat(z_start + (i+1) * st)
    ds.SliceLocation = pydicom.valuerep.DSfloat(z_start + i * st)
    pydicom.dcmwrite(os.path.join(mp, '2-{:02d}.dcm'.format(i+1)), ds)

# %%
mp = 'D:\\PhD\\Data\\TCGA_new_raw\\TCGA-10-0937\\pelvis'
ds_list = [pydicom.dcmread(os.path.join(mp, dcm)) for dcm in os.listdir(mp)]

# %%
for ds in ds_list:
    print(ds.ImagePositionPatient[2])

# %%

print(np.diff([ds.ImagePositionPatient[2] for ds in ds_list]))

# %%
rawp = 'D:\\PhD\\Data\\TCGA_new_TB'

for scan in os.listdir(rawp):
    
    content = os.listdir(os.path.join(rawp, scan))
    if len(content) == 2:
        print(scan, content)
        ds_list1 = [pydicom.dcmread(os.path.join(rawp, scan, content[0], dcm)) for dcm in 
                    os.listdir(os.path.join(rawp, scan, content[0])) if not dcm.startswith('TCGA')]
        ds_list2 = [pydicom.dcmread(os.path.join(rawp, scan, content[1], dcm)) for dcm in 
                    os.listdir(os.path.join(rawp, scan, content[1])) if not dcm.startswith('TCGA')]
        
        print_diff_attr(ds_list1, ds_list2)
