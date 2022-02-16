from ovseg.model.SLDSEnsemble import SLDSEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

w = [0.001, 0.01, 0.1][args.exp]

ens = SLDSEnsemble(val_fold=list(range(5)), data_name='OV04', preprocessed_name='SLDS',
                   model_name='U-Net5_{}'.format(w))
ens.eval_raw_dataset('BARTS', save_preds=True)


