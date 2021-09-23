from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.data.Dataset import low_res_ds_wrapper
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

p_name = 'bin_seg'
out_shape = [[24, 192, 192], #4
             [28, 224, 224], #2.3
             [36, 288, 288], #1.26
             [40, 320, 320]] #1

N_PROC = 4

all_M_values = np.arange(5, 21)
mean_dscs = []
for M in all_M_values:
    resp = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', 
                        'OV04', p_name, 'U-Net5_M_{}'.format(M), 'fold_0',
                        'BARTS_250_results.pkl')
    res = pickle.load(open(resp, 'rb'))
    mean_dscs.append(np.nanmean([res[key]['dice_1'] for key in res]))

M_list = [M for dsc, M in sorted(zip(mean_dscs, all_M_values))][args.exp:8:N_PROC]

scale = (np.array(out_shape[1]) / np.array(out_shape[-1])).tolist()

BARTS_low_res_ds = low_res_ds_wrapper('BARTS', scale)

for M in M_list:
    model = SegmentationModel(val_fold=0,
                              data_name='OV04',
                              preprocessed_name=p_name, 
                              model_name='U-Net5_M_{}'.format(M))

    if model.training.epochs_done < 500:
        model.training.train()
        model.eval_ds(BARTS_low_res_ds, 'BARTS_500', save_preds=False)
