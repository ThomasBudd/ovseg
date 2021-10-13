from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('vf', type=int)
args = parser.parse_args()


p_names = ['om_067', 'multiclass_1_2_9']
model_names = ['larger_res_encoder', 'U-Net5']

for p_name, model_name in zip(p_names, model_names):
    
    model = SegmentationModel(val_fold=args.vf,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name=model_name)
    model.eval_raw_data_npz('BARTS')

if args.vf <= 1:
    ens = SegmentationEnsemble(val_fold=list(range(5)),
                               data_name='OV04',
                               preprocessed_name=p_names[args.vf],
                               model_name=model_names[args.vf])
    ens.eval_raw_dataset('BARTS', save_preds=True)

