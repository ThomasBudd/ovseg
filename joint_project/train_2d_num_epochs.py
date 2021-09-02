from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("fold", type=int)
parser.add_argument("num_epochs", type=int)
parser.add_argument("--small", default=False, action='store_true')
args = parser.parse_args()

fp32 = False
if args.small:
    model_name = '2d_UNet_small_{}_v2'.format(args.num_epochs)
else:
    model_name = '2d_UNet_large_{}'.format(args.num_epochs)
    
model_params = get_model_params_2d_segmentation(fp32=fp32)

model_params['training']['num_epochs'] = args.num_epochs

if args.small:
    model_params['network']['filters'] = 8

preprocessed_name = 'pod_2d'


model = SegmentationModel(val_fold=args.fold,
                          data_name='OV04',
                          preprocessed_name=preprocessed_name,
                          model_name=model_name,
                          model_parameters=model_params)
model.save_model_parameters()
model.training.train()
model.eval_validation_set()
# we evaluate BARTS twice, once only with each model form each fold....
model.eval_raw_dataset('BARTS')
model.clean()

#... and a second time using ensembling
ens = SegmentationEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name=preprocessed_name,
                           model_name=model_name)
if ens.all_folds_complete():
    ens.eval_raw_dataset('BARTS', save_preds=True, save_plots=False)
