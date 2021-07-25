from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run", type=int)
parser.add_argument("--small", required=False, default=False, action='store_true')
args = parser.parse_args()


model_params = get_model_params_2d_segmentation()
model_params['network']['norm'] = 'inst'
model_params['network']['norm_params'] = {'affine': True, 'eps': 1e-2}
model_params['training']['stop_after_epochs'] = [250, 500, 750]
model_params['training']['opt_name'] = 'ADAM'
model_params['training']['opt_params'] = {'lr': 10**-4}
model_name = '2d_num_epochs_ADAM'
if args.small:
    model_params['network']['filters'] = 8
    model_name += '_small'


model = SegmentationModel(val_fold=5+args.run,
                          data_name='OV04',
                          model_name=model_name,
                          model_parameters=model_params,
                          preprocessed_name='pod_2d')

for num_epochs in [250, 500, 750, 1000]:
    model.training.train()
    
    model.eval_ds(model.data.val_ds, ds_name='validation_'+str(num_epochs), save_preds=False,
                  save_plots=False, merge_to_CV_results=True)