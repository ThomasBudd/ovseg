import os
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.ClaraWrappers import ClaraWrapperOvarian
from ovseg.data.Dataset import raw_Dataset
from ovseg.utils.io import read_nii
from tqdm import tqdm
import numpy as np

ds_name = 'BARTS'
data_name = 'OV04'
preprocessed_name = 'pod_om'
model_name = 'clara_model_no_tta'

path_to_clara_models = os.path.join(os.environ['OV_DATA_BASE'], 'clara_models')

# %%
ds = raw_Dataset(ds_name)
ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           preprocessed_name=preprocessed_name,
                           model_name=model_name)

pred_key = ens.models[0].pred_key

# %%
results = []
for i in tqdm(range(len(ds))):
    
    # load data_tpl    
    data_tpl = ds[i]
        
    # compute prediciton
    pred = ClaraWrapperOvarian(data_tpl, models=['pod_om'],
                               path_to_clara_models=path_to_clara_models)
    
    # save predictions move the z axis back to the front
    data_tpl[pred_key] = np.moveaxis(pred, -1, 0)
    results.append(ens.compute_error_metrics(data_tpl))
    ens.save_prediction(data_tpl, ds_name+'_clara')
    

# %%
print('DSC 1:')
print(np.nanmean([res['dice_1'] for res in results]))
print('DSC 9:')
print(np.nanmean([res['dice_9'] for res in results]))