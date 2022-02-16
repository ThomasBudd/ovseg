from ovseg.model.SegmentationEnsemble import SegmentationEnsemble

data_name = 'OV04'
model_name = 'U-Net4'
preprocessed_name = 'pod_om_08_5'

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=preprocessed_name)
ens.eval_raw_dataset('ApolloTCGA')

