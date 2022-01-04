from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

data_name = 'OV04'
preprocessed_name = 'pod_om_08_5'
model_name = 'sha_wd_0.98_1000_000'

model = SegmentationModel(val_fold=args.vf,
                          data_name=data_name,
                          model_name=model_name,
                          preprocessed_name=preprocessed_name)
model.eval_raw_data_npz('ApolloTCGA')

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=preprocessed_name)
ens.eval_raw_dataset('ApolloTCGA')