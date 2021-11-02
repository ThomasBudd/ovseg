from os import environ
from os.path import join
from ovseg.utils.io import load_pkl, save_pkl
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble

p_name='pod_om_08_25'
model_name='U-Net4_prg_lrn'
path_to_params = join(environ['OV_DATA_BASE'], 'trained_models', 
                      'OV04', p_name, model_name, 'model_parameters.pkl')

model_params = load_pkl(path_to_params)

model_params['postprocessing'] = {'apply_small_component_removing': True,
                                  'volume_thresholds': 10,
                                  'remove_2d_comps': True,
                                  'use_fill_holes_2d': True}

save_pkl(model_params, path_to_params)

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name='OV04',
                           model_name=model_name, 
                           preprocessed_name=p_name)

ens.eval_raw_dataset('ApolloTCGA_dcm')
