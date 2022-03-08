from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run", type=int)
parser.add_argument("--small", required=False, default=False, action='store_true')
args = parser.parse_args()

epochs = list(range(100, 2500, 100))

model_params = get_model_params_2d_segmentation()
model_params['network']['norm'] = 'inst'
model_params['network']['norm_params'] = {'affine': True, 'eps': 1e-2}
model_params['training']['stop_after_epochs'] = epochs
model_params['training']['num_epochs'] = 2500
model_params['training']['save_additional_weights_after_epochs'] = [2000]
model_params['training']['opt_name'] = 'ADAM'
model_params['training']['opt_params'] = {'lr': 10**-4}
model_name = '2d_num_epochs_ADAM_long'
if args.small:
    model_params['network']['filters'] = 8
    model_name += '_small'


model = SegmentationModel(val_fold=5+args.run,
                          data_name='OV04',
                          model_name=model_name,
                          model_parameters=model_params,
                          preprocessed_name='pod_2d')

for num_epochs in epochs + [2500]:
    model.training.train()
    
    model.eval_ds(model.data.val_ds, ds_name='validation_'+str(num_epochs), save_preds=False,
                  save_plots=False, merge_to_CV_results=True)

# %%
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

bp = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', 'OV04', 'pod_2d',
                  '2d_num_epochs_ADAM_long')

epochs = list(range(100, 2600, 100))
mean_dscs = []
for epoch in epochs:
    res = pickle.load(open(os.path.join(bp, 'validation_{}_CV_results.pkl'.format(epoch)), 'rb'))
    mean_dscs.append(np.mean([res[key]['dice_9'] for key in res]))

plt.plot(epochs, mean_dscs, 'b')
