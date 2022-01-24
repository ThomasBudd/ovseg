import os
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.ClaraWrappers import ClaraWrapperOvarian
from ovseg.data.Dataset import raw_Dataset
from ovseg.utils.io import read_nii
from tqdm import tqdm
import numpy as np

ds_name = 'ICON8_14_Derby_Burton'
data_name = 'OV04'
preprocessed_name = 'pod_om'
model_name = 'clara_model_no_tta'

path_to_clara_models = os.path.join(os.environ['OV_DATA_BASE'], 'clara_models')

ds = raw_Dataset(ds_name)
ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           preprocessed_name=preprocessed_name,
                           model_name=model_name)

pred_key = ens.models[0].pred_key

# %%

for i in tqdm(range(len(ds))):
    
    # load data_tpl    
    data_tpl = ds[i]
        
    # compute prediciton
    pred = ClaraWrapperOvarian(data_tpl, models=['pod_om'],
                               path_to_clara_models=path_to_clara_models)
    
    # save predictions move the z axis back to the front
    data_tpl[pred_key] = np.moveaxis(pred, -1, 0)
    ens.save_prediction(data_tpl, ds_name+'_clara')
    

# %%
ens.eval_raw_dataset(ds_name, save_preds=True)

# %% compare results

def DSC(p1, p2):
    return 200 * (np.sum(p1*p2) + 1e-5)/(np.sum(p1 + p2) + 1e-5)

def mc_DSC(p1, p2):
    
    return DSC((p1==1).astype(float),(p2==1).astype(float)) + DSC((p1==9).astype(float),(p2==9).astype(float))

predbp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions',
                      data_name, preprocessed_name, model_name, ds_name)

predp1 = predbp + '_ensemble_5_6_7'
predp2 = predbp + '_clara'

nii_files = [f for f in os.listdir(predp1) if f.endswith('.nii.gz')]

for nii_file in nii_files:
    
    pred1, _, _ = read_nii(os.path.join(predp1, nii_file))
    pred2, _, _ = read_nii(os.path.join(predp2, nii_file))
    
    dsc = mc_DSC(pred1, pred2) / 2
    
    print(nii_file.split('.')[0] + ': {:.3f}'.format(dsc))
    
    