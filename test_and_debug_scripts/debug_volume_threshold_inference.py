import numpy as np
from os import listdir, environ
from os.path import join
from ovseg.data.Dataset import raw_Dataset
from tqdm import tqdm
import pickle
from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from skimage.measure import label
from ovseg.utils.io import read_nii

data_name = 'OV04'
folder_name='cross_validation'
scans=None
image_folder=None
dcm_revers=True
dcm_names_dict=None
model_name = 'larger_res_encoder'
preprocessed_name = 'pod_067'
n_fg_classes = 1
params = pickle.load(open(join(environ['OV_DATA_BASE'],
                               'trained_models', 
                               data_name,
                               preprocessed_name,
                               model_name,
                               'model_parameters.pkl'), 'rb'))['preprocessing']
preprocessing = SegmentationPreprocessing(**params)


ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', data_name),
                 scans=scans,
                 image_folder=image_folder,
                 dcm_revers=dcm_revers,
                 dcm_names_dict=dcm_names_dict)
# path with predictions (should be stored as nibabel)
predp = join(environ['OV_DATA_BASE'], 'predictions', data_name, preprocessed_name,
             model_name, folder_name)
if n_fg_classes > 1:
    print('WARNING: finding optimal volume treshold is atm only implemented for '
          'single class problems.')
vols_delta_dsc = []
for i in tqdm(range(len(ds))):
    data_tpl = ds[i]
    # get ground truth and possible remove other labels from the image
    gt = (preprocessing.maybe_clean_label_from_data_tpl(data_tpl) > 0).astype(float)
    if not data_tpl['scan']+'.nii.gz' in listdir(predp):
        continue
    pred, spacing, _ = read_nii(join(predp, data_tpl['scan']+'.nii.gz'))
    pred = pred > 0
    # all connected components, but how to set the threshold for removing the too small ones?
    comps = label(pred)
    comps_list = np.array([comps == c for c in range(1, comps.max() + 1)])
    fac = np.prod(spacing)
    vols_list = np.array([np.sum(comps == c)*fac for c in range(1, comps.max() + 1)])
    comps_list = comps_list[vols_list.argsort()]
    vols_list = np.sort(vols_list)

    # this choice doesn't matter, will shift the total stats only by a constant
    dsc_old = 200*np.sum(pred * gt) / np.sum(gt + pred)
    for j, vol in enumerate(vols_list):
        # prepare to fill in all components that are greater than the current threshold
        pred_tr = np.zeros_like(pred, dtype=float)
        for k in range(j+1, len(vols_list)):
            pred_tr[comps_list[k]] = 1
        dsc_new = 200*np.sum(pred_tr * gt) / np.sum(gt + pred_tr)
        vols_delta_dsc.append((vol, dsc_new - dsc_old))
        dsc_old = dsc_new
vols_delta_dsc = np.array(sorted(vols_delta_dsc))
dscs_tr = np.cumsum(vols_delta_dsc[:, 1])
vol_tr = vols_delta_dsc[np.argmax(dscs_tr), 0]
print(vol_tr)
