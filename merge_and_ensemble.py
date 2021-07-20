from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.RegionfindingEnsemble import RegionfindingEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("preprocessed_name")
parser.add_argument("model_name")
parser.add_argument("--data_name", required=False, type=str, default='OV04')
parser.add_argument("--raw_data_name", required=False, type=str, default='BARTS')
args = parser.parse_args()

try:
    ens = SegmentationEnsemble(val_fold=list(range(5)), data_name=args.data_name,
                               preprocessed_name=args.preprocessed_name,
                               model_name=args.model_name)
except TypeError:
    print('Caught TypeError, is the model a Regionfinding model?')
    ens = RegionfindingEnsemble(val_fold=list(range(5)), data_name=args.data_name,
                               preprocessed_name=args.preprocessed_name,
                               model_name=args.model_name)
    

if not ens.all_folds_complete():
    raise FileNotFoundError('Some folds seem to be uninifished')

ens.models[0]._merge_results_to_CV()
ens.eval_raw_dataset(args.raw_data_name, save_preds=True)