import os
import pydicom
import numpy as np
from ovseg.utils.io import save_dcmrt_from_data_tpl, read_dcms
from shutil import copytree, rmtree
from tqdm import tqdm

data_path = 'D:\\PhD\\Data\\TCGA_new_RW'

fol_list = []
for root, dirs, files in os.walk(data_path):
    if len(dirs) == 2:
        fol_list.append(root)

fol = fol_list[0]
# %%
def read_imagess(fol):
    
    p1 = os.path.join(fol, 'abdomen')
    p2 = os.path.join(fol, 'pelvis')
    
    ds_list1 = [pydicom.dcmread(os.path.join(p1, dcm)) for dcm in os.listdir(p1)
                if not dcm.startswith('TCGA') and dcm.endswith('.dcm')]
    ds_list2 = [pydicom.dcmread(os.path.join(p2, dcm)) for dcm in os.listdir(p2)
                if not dcm.startswith('TCGA') and dcm.endswith('.dcm')]
    
    print('read {} and {} dcms'.format(len(ds_list1), len(ds_list2)))
    print_diff_attr(ds_list1, ds_list2)
    return ds_list1, ds_list2

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

def modify_ipp(ds_list1, ds_list2):
    st = np.median(np.diff([ds.ImagePositionPatient[2] for ds in ds_list1]))
    x, y = ds_list1[0].ImagePositionPatient[0], ds_list1[0].ImagePositionPatient[1]
    z_start = float(ds_list1[-1].ImagePositionPatient[2])
    ds_list2_new = []
    for i, ds in enumerate(ds_list2):
        ds.ImagePositionPatient[0] = x
        ds.ImagePositionPatient[1] = y
        ds.ImagePositionPatient[2] = pydicom.valuerep.DSfloat(z_start + (i+1) * st)
        if hasattr(ds, 'SliceLocation'):
           ds.SliceLocation = pydicom.valuerep.DSfloat(z_start + i * st)
        ds_list2_new.append(ds)
    return ds_list2_new

def save_merged_images(ds_list1, ds_list2, target_path):
    
    for i, ds in enumerate(ds_list1):
        pydicom.dcmwrite(os.path.join(target_path, '1-{:03d}.dcm'.format(i+1)), ds)
        
    for i, ds in enumerate(ds_list2):
        pydicom.dcmwrite(os.path.join(target_path, '1-{:03d}.dcm'.format(i+1+len(ds_list1))),
                         ds)

def read_labels(fol):
    
    p1 = os.path.join(fol, 'abdomen')
    p2 = os.path.join(fol, 'pelvis')
    
    data_tpl1 = read_dcms(p1)
    
    if 'label' in data_tpl1:
        label1 = data_tpl1['label']
    else:
        label1 = np.zeros_like(data_tpl1['image'])
    
    data_tpl2 = read_dcms(p2)
    
    if 'label' in data_tpl2:
        label2 = data_tpl2['label']
    else:
        label2 = np.zeros_like(data_tpl2['image'])
    
    data_tpl1['new_label'] = np.concatenate([label1, label2])
    
    return data_tpl1
    
def read_and_merge(fol):
    root, scan = os.path.split(fol)
    data_base, dataset_name = os.path.split(root)
    target_path = os.path.join(data_base, dataset_name+'_merged', scan)
    rmtree(target_path)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    ds_list1, ds_list2 = read_imagess(fol)
    ds_list2_new = modify_ipp(ds_list1, ds_list2)
    save_merged_images(ds_list1, ds_list2_new, target_path)

    data_tpl = read_labels(fol)
    data_tpl['raw_image_file'] = target_path
    out_file = os.path.join(target_path, scan+'_'+dataset_name[-2:]+'.dcm')
    save_dcmrt_from_data_tpl(data_tpl, out_file, key='new_label')

# %% merge for folders with two subfolders

for data_path in ['D:\\PhD\\Data\\TCGA_new_TB']:#['D:\\PhD\\Data\\TCGA_new_RW', 'D:\\PhD\\Data\\TCGA_new_TB']:

    fol_list = []
    for root, dirs, files in os.walk(data_path):
        if len(dirs) == 2:
            fol_list.append(root)

    for fol in fol_list:
        read_and_merge(fol)

# %% copy the other scans with only one folder

for data_path in ['D:\\PhD\\Data\\TCGA_new_TB']:#['D:\\PhD\\Data\\TCGA_new_RW', 'D:\\PhD\\Data\\TCGA_new_TB']:
    tarp = data_path + '_merged'
    
    for scan in tqdm(os.listdir(data_path)):
        if scan not in os.listdir(tarp):
            print('Copy', scan)
            copytree(os.path.join(data_path, scan),
                     os.path.join(tarp, scan))