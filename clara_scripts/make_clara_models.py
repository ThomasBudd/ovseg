import os
from ovseg.utils.io import load_pkl, save_pkl, save_txt
from shutil import copy


source_path = os.path.join(os.environ['OV_DATA_BASE'],
                           'trained_models', 
                           'ApolloTCGA_BARTS_OV04')

target_path = os.path.join(os.environ['OV_DATA_BASE'],
                           'clara_models')


for model in os.listdir(source_path):
    print(model)
    model_target_path = os.path.join(target_path, model)
    
    if not os.path.exists(model_target_path):
        os.makedirs(model_target_path)
    
    model_params = load_pkl(os.path.join(source_path, model, 'clara_model', 'model_parameters.pkl'))
    model_params['prediction']['mode'] = 'simple'
    
    save_pkl(model_params, os.path.join(model_target_path, 'model_parameters.pkl'))
    save_txt(model_params, os.path.join(model_target_path, 'model_parameters.txt'))
    
    for i, vf in enumerate([5,6,7]):
        copy(os.path.join(source_path, model, 'clara_model',f'fold_{vf}', 'network_weights'),
             os.path.join(model_target_path, f'network_weights_{i}'))