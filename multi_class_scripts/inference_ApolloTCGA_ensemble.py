from os import environ
from os.path import join
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.SegmentationModel import SegmentationModel

p_name='pod_om_08_25'
model_name='U-Net4_prg_lrn'


ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name='OV04',
                           model_name=model_name, 
                           preprocessed_name=p_name)

ens.eval_raw_dataset('ApolloTCGA_dcm')

model = SegmentationEnsemble(val_fold=5,
                             data_name='OV04',
                             model_name=model_name, 
                             preprocessed_name=p_name)

model.eval_raw_dataset('ApolloTCGA_dcm')
