from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.model.RegionfindingEnsemble import RegionfindingEnsemble

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

w = [2/3, 2/5, 2/9, 2/17, 2/33][args.vf]

p_name = 'lymph_reg'

ens = RegionfindingEnsemble(val_fold=list(range(5)), 
                            data_name='OV04',
                            preprocessed_name=p_name,
                            model_name='regfinding_'+str(w))


ens.eval_raw_dataset('BARTS', save_preds=True)