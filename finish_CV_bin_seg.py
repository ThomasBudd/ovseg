from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=int)
args = parser.parse_args()

p_name = 'bin_seg'
vf_list = [[1, 2], [3, 4]][args.exp]
M_list = list(range(5, 21))

def check_if_all_finished():
    not_finished = 0
    for M in M_list:
        p = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', 'OV04', p_name, 
                         'U-Net5_M_{}'.format(M), 'fold_0', 'BARTS_1000_results.pkl')
        if not os.path.exists(p):
            not_finished += 1

    if not_finished > 0:
        print('{} trainings not finished...'.format(not_finished))
        return False
    else:
        print('all finished!')
        return True


s = 0
while not check_if_all_finished():
    sleep(60)
    s += 1
    print('Slept {} minutes'.format(s))

for ep in [250, 500, 750, 1000]:
    mdscs = []
    for M in M_list:
        p = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', 'OV04', p_name, 
                         'U-Net5_M_{}'.format(M), 'fold_0', 'BARTS_{}_results.pkl'.format(M))
        
        res = pickle.load(open(p, 'rb'))
        mdscs.append(np.nanmean([res[key]['dice_1'] for key in res]))

    plt.plot(M_list, mdscs)

plt.legend([250, 500, 750, 1000])
plt.savefig(os.path.join(os.environ['OV_DATA_BASE'], 'tune_M_bin_seg.png'))

M = M_list[np.argmax(mdscs)]

for vf in vf_list:
    model = SegmentationModel(val_fold=vf,
                              data_name='OV04',
                              preprocessed_name=p_name, 
                              model_name='U-Net5_M_{}'.format(M))

    while model.training.epochs_done < 1000:
        model.training.train()

    model.eval_raw_data_npz('BARTS')
