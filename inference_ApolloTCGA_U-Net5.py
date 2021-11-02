from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

model = SegmentationModel(val_fold=args.vf,
                          data_name='OV04',
                          preprocessed_name='multiclass_1_9',
                          model_name='U-Net5_new_sampling',
                          is_inference_only=True)
model.eval_raw_data_npz('ApolloTCGA')

if args.vf == 0:
    ens = SegmentationEnsemble(val_fold=list(range(5)),
                               data_name='OV04',
                               preprocessed_name='multiclass_1_9',
                               model_name='U-Net5_new_sampling')
    ens.eval_raw_data_npz('ApolloTCGA')