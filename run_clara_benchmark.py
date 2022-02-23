from ovseg.model.ClaraWrappers import ClaraWrapperOvarian
import os
from ovseg.data.Dataset import raw_Dataset
from ovseg.util.io import save_nii_from_data_tpl
from tqdm import tqdm

path_to_clara_models = os.path.join(os.environ['OV_DATA_BASE'],
                                    'clara_models')

predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 
                     'clara_benchmark')

models_list = ['abdominal_lesions', 'lymph_nodes', 'pod_om']

for model in models_list:
    p = os.path.join(predp, model)
    if not os.path.exists(p):
        os.makedirs(p)

ds = raw_Dataset('TCGA_clara_test')

for data_tpl in tqdm(ds):
    
    for model in models_list:
        
        # evaluate wrapper
        pred = ClaraWrapperOvarian(data_tpl, [model], path_to_clara_models)
        
        # store in data_tpl
        pred_key = f'pred_{model}'
        data_tpl[pred_key] = pred
        
        # export as nifti
        out_file = os.path.join(predp, model, data_tpl['scan'])
        save_nii_from_data_tpl(data_tpl, out_file, pred_key)        
        
