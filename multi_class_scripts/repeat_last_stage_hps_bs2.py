from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("i", type=int)
args = parser.parse_args()


ind = [k for k in [0, 0, 3, 3, 4, 4, 6, 6]][args.i]
vf = [vf for vf in 4*[6,7]][args.i]

data_name = 'OV04'
preprocessed_name = 'pod_067'
model_name = 'hpo_bs_2_in_1000_{:03d}'.format(ind)

model = SegmentationModel(val_fold=vf,
                          data_name=data_name,
                          preprocessed_name=preprocessed_name,
                          model_name=model_name)
model.training.train()

ens = SegmentationEnsemble(val_fold=[5,6,7],
                          data_name=data_name,
                          preprocessed_name=preprocessed_name,
                          model_name=model_name)

ens.wait_until_all_folds_complete()

ens.eval_raw_dataset('BARTS')
