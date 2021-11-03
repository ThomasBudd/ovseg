from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

p_name = 'pod_067'
model_name = 'larger_res_encoder'
data_name='OV04'
model = SegmentationModel(val_fold=args.vf,
                          data_name=data_name,
                          preprocessed_name=p_name, 
                          model_name=model_name)
model.training.train()
model.eval_raw_data_npz('BARTS')

ens = SegmentationEnsemble(val_fold=[5,6,7],
                           data_name=data_name,
                           model_name=model_name,
                           preprocessed_name=p_name)

while not ens.all_folds_complete():
    sleep(60)

ens.eval_raw_dataset('BARTS')