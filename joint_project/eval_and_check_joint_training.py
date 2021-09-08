import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import join
import pickle

mbp = 'D:\\PhD\\Data\\ov_data_base\\trained_models\\OV04\\pod_2d'
C_list = [1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1]
#font size
fs = 18

# %%
for k, dose in enumerate(['full', 'quater']):
    plt.subplot(1, 2, k+1)
    plt.title(dose, fontdict = {'fontsize' : fs})
    joint_full = [join(mbp, 'joint_rest_seg_{}_{}'.format(dose, c)) for c in C_list]
    
    mean_dscs = []
    not_finished = []
    
    for i, cvp in enumerate(joint_full):
    
        training_not_finished = False
        
        mean_dsc = 0
    
        for j in range(5,8):
            trp = join(cvp, 'fold_{}'.format(j))
            tr_attr = pickle.load(open(join(trp, 'attribute_checkpoint.pkl'), 'rb'))
            if tr_attr['epochs_done'] < 1000:
                training_not_finished = True
                print('training not finished for c={}, fold={}'.format(C_list[i], j))
            res = pickle.load(open(join(trp, 'validation_results.pkl'), 'rb'))
            mean_dsc += np.mean([res[key]['dice_9'] for key in res])
    
        mean_dsc /= 3
        mean_dscs.append(mean_dsc)
    
        if training_not_finished:
            not_finished.append(i)
    
    plt.plot(C_list, mean_dscs, 'b')
    
    # get the sequential DSC
    seq_p = join(mbp, '2d_sequential_new_{}'.format(dose))
    res = pickle.load(open(join(seq_p, 'validation_CV_results.pkl'), 'rb'))
    mean_seq_dsc = np.mean([res[key]['dice_9'] for key in res])
    plt.plot([0.1, 1], [mean_seq_dsc, mean_seq_dsc], 'r')
    for i in range(len(C_list)):
        if i in not_finished:
            plt.plot(C_list[i], mean_dsc[i]+0.1, 'b*')
    
    if k == 0:
        plt.ylim([30, 45])
    else:
        plt.ylim([25, 40])
        plt.legend(['joint', 'sequential'], loc='lower right', prop={'size': fs})

    plt.xlabel('C', fontsize=fs)
    plt.ylabel('mean DSC', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
# %%
plt.figure()
for k, dose in enumerate(['full', 'quater']):
    plt.subplot(1, 2, k+1)
    plt.title(dose, fontdict = {'fontsize' : fs})
    joint_full = [join(mbp, 'joint_rest_seg_refine_{}_{}'.format(dose, c)) for c in C_list]
    
    mean_dscs = []
    not_finished = []
    
    for i, cvp in enumerate(joint_full):
    
        training_not_finished = False
        
        mean_dsc = 0
    
        for j in range(5,8):
            trp = join(cvp, 'fold_{}'.format(j))
            tr_attr = pickle.load(open(join(trp, 'attribute_checkpoint.pkl'), 'rb'))
            if tr_attr['epochs_done'] < 500:
                training_not_finished = True
                print('training not finished for c={}, fold={}, dose={}'.format(C_list[i], j, dose))
            try:
                res = pickle.load(open(join(trp, 'validation_results.pkl'), 'rb'))
                mean_dsc += np.mean([res[key]['dice_9'] for key in res])
            except FileNotFoundError:
                print('results not found!')
                mean_dsc += -1
    
        mean_dsc /= 3
        mean_dscs.append(mean_dsc)
    
        if training_not_finished:
            not_finished.append(i)
    
    plt.plot(C_list, mean_dscs, 'b')
    
    # get the sequential DSC
    seq_p = join(mbp, '2d_sequential_refine_{}'.format(dose))
    res_list = [pickle.load(open(join(seq_p, 'fold_{}'.format(f),
                                      'validation_results.pkl'), 'rb'))
                for f in range(5,8)]
    mean_seq_dsc = np.mean([np.mean([res[key]['dice_9'] for key in res]) for res in res_list])
    plt.plot([0.1, 1], [mean_seq_dsc, mean_seq_dsc], 'r')
    for i in range(len(C_list)):
        if i in not_finished:
            plt.plot(C_list[i], mean_dscs[i]+0.1, 'b*')
    
    if k == 0:
        plt.ylim([30, 45])
    else:
        plt.ylim([30, 45])
        plt.legend(['joint', 'sequential'], loc='lower right', prop={'size': fs})

    plt.xlabel('C', fontsize=fs)
    plt.ylabel('mean DSC', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)