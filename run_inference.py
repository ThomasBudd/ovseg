import os
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.ClaraWrappers import ClaraWrapperOvarian
from ovseg.data.Dataset import raw_Dataset
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("tst_data")
# add all the names of the labled training data sets as trn_data
parser.add_argument("--trn_data", default=['OV04', 'BARTS', 'ApolloTCGA'], nargs='+')
parser.add_argument("--models", default=['pod_om', 'abdominal_lesions','lymph_nodes'], nargs='+')

args = parser.parse_args()

vf = args.vf
trn_data = args.trn_data
data_name = '_'.join(sorted(trn_data))
models = args.models
tst_data = args.tst_data

# change the model name when using other hyper-paramters
model_name = 'clara_model'
# if you store the weights and model parameters somewhere else, please change
# this path
path_to_clara_models = os.path.join(os.environ['OV_DATA_BASE'], 'clara_models')

# %% get ensemble and dataset
ds = raw_Dataset(tst_data)
ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           preprocessed_name=models[0],
                           model_name=model_name)

pred_key = ens.models[0].pred_key

# %% iterate over the dataset and save predictions

for i in tqdm(range(len(ds))):
    
    # load data_tpl    
    data_tpl = ds[i]
        
    # compute prediciton
    pred = ClaraWrapperOvarian(data_tpl, 
                               models=models,
                               path_to_clara_models=path_to_clara_models)
    
    # save predictions move the z axis back to the front
    data_tpl[pred_key] = np.moveaxis(pred, -1, 0)
    ens.save_prediction(data_tpl, tst_data)

print('Inference done!')
print('Saved predictions can be found here:')
pred_folder = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', data_name,
                           models[0], model_name, tst_data)
print(pred_folder)

