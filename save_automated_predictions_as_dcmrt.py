# from ovseg.utils.io import read_dcms
from ovseg.utils.io import _is_im_dcm_ds, _is_roi_dcm_ds
import pydicom
import numpy as np
from os import listdir, environ
from os.path import join, basename
import nibabel as nib
import pickle
from tqdm import tqdm
from rt_utils import RTStructBuilder

dcmp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS_dcm')
niip = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS')
predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04')

pred_folders = [['pod_067', 'larger_res_encoder'], ['om_08', 'res_encoder_no_prg_lrn'],
                ['lesions_center', 'res_encoder_no_prg_lrn'],
                ['lesions_upper', 'res_encoder_no_prg_lrn'],
                ['lesions_lymphnodes', 'res_encoder_no_prg_lrn']]

colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 165, 0], [0, 128, 128],
          [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 165, 0], [0, 128, 128]]
names = ['9-auto', '1-auto', '5,6,7-auto', '2,3,4-auto', '13,14,15-auto']

all_cases = listdir(join(niip, 'labels'))
data_info = pickle.load(open(join(niip, 'data_info.pkl'), 'rb'))

# %%
def read_dcms(dcm_folder):
    dcms = [join(dcm_folder, dcm) for dcm in listdir(dcm_folder)]
    dcms.sort()
    imdss = []
    roidss = []
    roidcms = []
    for dcm in dcms:
        ds = pydicom.dcmread(dcm)
        if _is_im_dcm_ds(ds):
            imdss.append(ds)
        elif _is_roi_dcm_ds(ds):
            roidss.append(ds)
            roidcms.append(dcm)
        else:
            raise TypeError(dcm + ' is neither image nor roi dcm.')
    if len(roidss) > 1:
        raise FileExistsError('Found multiple ROI dcms in folder '+dcm_folder+'. '
                              'Make sure that at most one ROI dcm file is in each folder.')
    z_im = [imds.ImagePositionPatient[2] for imds in imdss]
    imdss = [ds for _, ds in sorted(zip(z_im, imdss), reverse=True)]
    z_im = [imds.ImagePositionPatient[2] for imds in imdss]
    try:
        z_loc = [imds.SliceLocation for imds in imdss]
        if np.all(np.array(z_im) == np.array(z_loc)):
            print('ImagePositionPatient[2] match SliceLocation')
        elif np.all(np.array(z_im) == -1*np.array(z_loc)):
            print('ImagePositionPatient[2] with sign flip SliceLocation')
        else:
            print('ImagePositionPatient[2] dont match SliceLocation')
            
    except Exception:
        print('no slice location found')
    return z_im, roidcms[0]

# %%
case = all_cases[3]

names = ['9-auto', '1-auto', '5,6,7-auto', '2,3,4-auto', '13,14,15-auto']

for case in tqdm(all_cases):
    preds = [nib.load(join(predp, pn, mn, 'BARTS_ensemble_0_1_2_3_4', case)).get_fdata()
             for pn, mn in pred_folders]
    di = data_info[case[5:8]]
    dcm_folder = join(dcmp, basename(di['scan']))
    
    z_im, dcmrt = read_dcms(dcm_folder)
    if -1*float(np.diff(z_im)[0]) != 5:
        print('skipp!')
        continue
    
    rtstruct = RTStructBuilder.create_from(
      dicom_series_path=dcm_folder, 
      rt_struct_path=dcmrt
    )
    
    # Add ROI. This is the same as the above example.
    for pred, color, name in zip(preds, colors, names):
        if pred.max() == 0:
            continue
        rtstruct.add_roi(
          mask=pred>0, 
          color=color, 
          name=name
        )
        
        rtstruct.ds.StructureSetROISequence[-1].ROIGenerationAlgorithm = "AUTOMATIC"
        
        cs = rtstruct.ds.ROIContourSequence[-1].ContourSequence
        
        # for d in cs:
        #     for i in range(2, len(d.ContourData), 3):
        #         print(d.ContourData[i])
                # if float(d.ContourData[i]) > 0:
                    # d.ContourData[i] = pydicom.valuerep.DSfloat(-1*float(d.ContourData[i]))
    
    rtstruct.save(join(predp, 'dcm_rt_single_class', basename(di['scan'])))

# %%

pred_folder = join(predp, 'multiclass_cascade_08', 'res_encoder', 'BARTS_ensemble_0_1_2_3_4')
classes = [1, 9, 2, 3, 4, 5, 6, 7, 13, 14, 15]
names = [str(c)+'-auto' for c in classes]

for case in tqdm(all_cases):
    pred = nib.load(join(pred_folder, case)).get_fdata()
    di = data_info[case[5:8]]
    dcm_folder = join(dcmp, basename(di['scan']))
    
    z_im, dcmrt = read_dcms(dcm_folder)
    if -1*float(np.diff(z_im)[0]) != 5:
        print('skipp!')
        continue
    
    rtstruct = RTStructBuilder.create_from(
      dicom_series_path=dcm_folder, 
      rt_struct_path=dcmrt
    )
    
    # Add ROI. This is the same as the above example.
    for i, (color, name) in enumerate(zip(colors, names)):
        mask = pred == i+1
        if mask.max() == 0:
            continue
        rtstruct.add_roi(
          mask=mask[...,::-1], 
          color=color, 
          name=name
        )
        
        rtstruct.ds.StructureSetROISequence[-1].ROIGenerationAlgorithm = "AUTOMATIC"
        
        cs = rtstruct.ds.ROIContourSequence[-1].ContourSequence
        
        for d in cs:
            for i in range(2, len(d.ContourData), 3):
                if float(d.ContourData[i]) > 0:
                    d.ContourData[i] = pydicom.valuerep.DSfloat(-1*float(d.ContourData[i]))
    
    rtstruct.save(join(predp, 'dcm_rt_multiclass', basename(di['scan'])))