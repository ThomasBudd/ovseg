import os
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.ClaraWrappers import ClaraWrapperOvarian, preprocess_dynamic_z_spacing
from ovseg.data.Dataset import raw_Dataset
from ovseg.utils.io import read_nii, load_pkl
from tqdm import tqdm
import numpy as np

ds_name = 'ICON8_14_Derby_Burton'
data_name = 'OV04'
preprocessed_name = 'pod_om'
model_name = 'clara_model_no_tta'

path_to_clara_models = os.path.join(os.environ['OV_DATA_BASE'], 'clara_models')

# %%
ds = raw_Dataset(ds_name)

data_tpl = ds[0]

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           preprocessed_name=preprocessed_name,
                           model_name=model_name)

pred_key = ens.models[0].pred_key

# %% compare the preprocessing
path_to_model_params = os.path.join(path_to_clara_models,'pod_om', 'model_parameters.pkl')
prep_params = load_pkl(path_to_model_params)['preprocessing']

clara_prep = preprocess_dynamic_z_spacing(data_tpl, prep_params)[0]

ens_prep = ens.preprocessing(data_tpl, preprocess_only_im=True)

ad = (clara_prep - ens_prep).abs()[0]

print(ad.mean().item())
print(ad.max().item())