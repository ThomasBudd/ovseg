import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

p_name = 'SLDS_reg_expert_0.01'
model_name = 'U-Net2'

val_names = ['validation_{}_results.pkl'.format(i) for i in range(100, 1000, 100)] + ['validation_results.pkl']
barts_names = ['BARTS_{}_results.pkl'.format(i) for i in range(100, 1000, 100)] + ['BARTS_results.pkl']

epochs = list(range(100,1100, 100))
val_dscs = []
barts_dscs = []

for barts_name, val_name in zip(barts_names, val_names):
    val_dsc = 0
    barts_dsc = 0
    for fold in range(5):
        val_res = pickle.load(os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', 'OV04',
                                           p_name, model_name, 'fold_{}'.format(fold),
                                           val_name))
        val_dsc += np.nanmean([val_res[key]['dice_2'] for key in val_res])
        barts_res = pickle.load(os.path.join(os.environ['OV_DATA_BASE'], 'trained_models', 'OV04',
                                             p_name, model_name, 'fold_{}'.format(fold),
                                             barts_name))
        barts_dsc += np.nanmean([barts_res[key]['dice_2'] for key in barts_res])
    val_dscs.append(val_dsc)
    barts_dscs.append(barts_dsc)

plt.plot(epochs, val_dscs, 'b')
plt.plot(epochs, barts_dscs, 'r')
plt.legend(['validation', 'barts'])
plt.title('training epochs vs. DSC')
plt.savefig(os.path.join(os.environ['OV_DATA_BASE'], 'U-Net2_epochs_vs_dsc.png'))