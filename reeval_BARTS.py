from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run", type=int)
args = parser.parse_args()

p_names = ['pod_067', 'pod_067', 'om_067', 'pod_08', 'pod_10', 'pod_12', 'om_10', 'om_12']
model_names = ['larger_res_encoder', 'res_encoder', 'res_encoder_no_prg_lrn', 'res_encoder',
               'res_encoder', 'res_encoder', 'res_encoder_no_prg_lrn', 'res_encoder_no_prg_lrn']


# %%
ens = SegmentationEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name=p_names[args.run],
                           model_name=model_names[args.run])
ens.eval_raw_dataset('BARTS', save_preds=True)
ens.clean()
# %%
ens = SegmentationEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name=p_names[args.run+4],
                           model_name=model_names[args.run+4])
ens.eval_raw_dataset('BARTS', save_preds=True)
