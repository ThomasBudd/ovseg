from ovseg.model.RegionfindingModel import RegionfindingModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

w = 2/33

p_name = 'lymph_reg'

model = RegionfindingModel(val_fold=args.vf,
                           data_name='OV04',
                           preprocessed_name=p_name,
                           model_name='regfinding_'+str(w))
model.eval_raw_data_npz('BARTS')