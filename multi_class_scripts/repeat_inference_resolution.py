from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

res_list = [['067', '12'], ['08', '10'], ['10', '08'], ['12', '067']][args.exp]
pref_list = ['pod_', 'om_']
model_names = ['res_encoder', 'res_encoder_no_prg_lrn']

p_names = [pref + res for pref, res in zip(pref_list, res_list)]

for p_name, model_name in zip(p_names, model_names):
    
    ens = SegmentationEnsemble(val_fold=list(range(5)),
                               data_name='OV04',
                               model_name=model_name, 
                               preprocessed_name=p_name)
    
    ens.eval_raw_dataset('BARTS', save_preds=True, force_evaluation=True)

