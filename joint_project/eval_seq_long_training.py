import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

cvp = 'D:\\PhD\\Data\\ov_data_base\\trained_models\\OV04\\pod_2d\\2d_num_epochs_ADAM_long'
pkl_files = [f for f in os.listdir(cvp) if f.startswith('v') and f.endswith('pkl') and f[11].isdigit()]
fs= 18
epochs = []
cvdcs = []
for f in pkl_files:
    res = pickle.load(open(os.path.join(cvp, f), 'rb'))
    cvdcs.append(np.mean([res[key]['dice_9'] for key in res]))
    epochs.append(int(f.split('_')[1]))
    
cvdcs = [d for e, d in sorted(zip(epochs, cvdcs))]
epochs = sorted(epochs)

plt.plot(epochs, cvdcs)
plt.title('Segmentation pretraining', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.xlabel('Num epochs (250 batches)', fontsize=fs)
plt.ylabel('Mean val DSC', fontsize=fs)


# %%
tr_attr = pickle.load(open(os.path.join(cvp, 'fold_5', 'attribute_checkpoint.pkl'), 'rb'))
val_losses = tr_attr['val_losses']
print(np.argmin(val_losses))