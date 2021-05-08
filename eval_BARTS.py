from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("i")

args = parser.parse_args()

if int(args.i) == 0:
    model_name = 'nnUNet_benchmark'
    p_name = 'pod_half'
elif int(args.i) == 1:
    model_name = 'prg_trn'
    p_name = 'pod_half'
elif int(args.i) == 2:
    model_name = 'prg_trn'
    p_name = 'om_half'
elif int(args.i) == 3:
    model_name = 'prg_trn'
    p_name = 'om_half'

    ens = SegmentationEnsemble(val_fold=list(range(5)),
                               data_name='OV04',
                               preprocessed_name=p_name,
                               model_name=model_name)
    
    if ens.all_folds_complete():
        ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False,
                             force_evaluation=True)
