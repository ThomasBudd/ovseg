from ovseg.model.SegmentationEnsembleV2 import SegmentationEnsembleV2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("i", type=int)
parser.add_argument("n", type=int)
args = parser.parse_args()

w_list = list(range(-3,4))[args.i::args.n]

data_name = 'kits21_trn'
preprocessed_name = 'disease_3_1'

for w in w_list:
    
    model_name = f'UQ_calibrated_{w:.2f}'
    ens = SegmentationEnsembleV2(val_fold=[0,1,2],
                                 model_name=model_name,
                                 data_name=data_name,
                                 preprocessed_name=preprocessed_name)
    ens.eval_raw_dataset('kits21_tst')