from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
import argparse

is_small = True

parser = argparse.ArgumentParser()
parser.add_argument("run", type=int)
args = parser.parse_args()


model_params = get_model_params_2d_segmentation(fp32=True)
model_params['network']['norm'] = 'inst'
model_params['network']['norm_params'] = {'affine': True, 'eps': 1e-2}
model_params['training']['stop_after_epochs'] = [250, 500, 750]
model_name = '2d_num_epochs'
if is_small:
    model_params['network']['filters'] = 8
    model_name += '_small'


model = SegmentationModel(val_fold=5+args.run,
                          data_name='OV04',
                          model_name=model_name,
                          model_parameters=model_params,
                          preprocessed_name='pod_2d')

for num_epochs in range([250, 500, 750, 1000]):
    model.training.train()
    
    model.eval_ds(model.val_ds, ds_name='validation_'+str(num_epochs), save_preds=False,
                  save_plots=False, merge_to_CV_results=True)