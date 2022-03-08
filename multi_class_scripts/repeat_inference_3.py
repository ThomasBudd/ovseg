from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.utils.io import load_pkl, save_pkl, read_dcms
import os
from time import sleep

data_name = 'OV04'
p_name = 'pod_om_08_25'
model_name = 'U-Net4_prg_lrn'

path_to_params = os.path.join(os.environ['OV_DATA_BASE'],
                              'trained_models',
                              data_name,
                              p_name,
                              model_name,
                              'model_parameters.pkl')
mp = load_pkl(path_to_params)

print(mp['postprocessing'])
sleep(5)
mp['postprocessing'] = {'apply_small_component_removing': True,
                        'volume_thresholds': 10,
                        'remove_2d_comps': True,
                        'remove_comps_by_volume': False,
                        'mask_with_reg': False,
                        'use_fill_holes_2d': True,
                        'apply_morph_cleaning': False}
save_pkl(mp, path_to_params)

ens = SegmentationEnsemble(val_fold=list(range(5)), data_name=data_name,
                           preprocessed_name=p_name,
                           model_name=model_name)

path_to_dcms = os.path.join(os.environ['OV_DATA_BASE'],
                            'raw_data',
                            'TCGA_new',
                            'TCGA-13-1509')
data_tpl = read_dcms(path_to_dcms)

pred = ens(data_tpl)

ens.save_prediction(data_tpl, 'TCGA_new')