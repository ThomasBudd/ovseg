import os
import shutil

bp = os.path.join(os.environ['OV_DATA_BASE'],
                  'trained_models',
                  'kits21')
tp = os.path.join(os.environ['OV_DATA_BASE'],
                  'trained_models',
                  'kits21_for_Bill')

files = ['model_parameters.txt', 'model_parameters.pkl',
         'validation_CV_results.pkl', 'validation_CV_results.txt']

for p_name in os.listdir(bp):
    
    for model_name in os.listdir(os.path.join(bp, p_name)):
        
        mp = os.path.join(tp, p_name, model_name)
        
        if not os.path.exists(mp):
            os.makedirs(mp)
        
        
        for file in files:
            if os.path.exists(os.path.join(bp, p_name, model_name, file)):
                shutil.copy(os.path.join(bp, p_name, model_name, file),
                            os.path.join(mp, file))


