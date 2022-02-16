from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.utils.io import read_dcms
import os

ens = SegmentationEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name='pod_om_08_25',
                           model_name='U-Net4_prg_lrn')

for fol in ['upper', 'lower']:
    
    rawp = os.path.join(os.environ['OV_DATA_BASE'],
                        'raw_data',
                        'TCGA_jump')
    
    data_tpl = read_dcms(os.path.join(rawp, fol))
    data_tpl['scan'] = fol
    
    pred = ens(data_tpl)
    ens.save_prediction(data_tpl, folder_name='TCGA_jump', filename=fol)
    
