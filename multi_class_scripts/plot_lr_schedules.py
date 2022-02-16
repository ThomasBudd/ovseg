import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots',
                     'lr_schedule')
if not os.path.exists(plotp):
    os.mkdir(plotp)

N = 250000
x = np.arange(N)

lr1 = 0.01 * (1- x / N)**0.9

lr2 = np.concatenate([0.02 * x[:N//20]/(N/20), 0.02 * np.cos(np.pi/2 * (x[N//20:]-N//20)/(N*0.95))])

plt.title('Learning rate schedules')
plt.plot(x, lr1, 'b', x, lr2, 'r')
plt.ylabel('Learning rate')
plt.xlabel('Batches')
plt.savefig(os.path.join(plotp, 'lr_schedules'), bbox_inches='tight')
plt.close()

trp = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models',
                   'OV04', 'pod_half', 'lr_schedule_0.02', 'fold_5',
                   'attribute_checkpoint.pkl')

attr = pickle.load(open(trp, 'rb'))

plt.title('Training progress')
plt.plot(np.arange(0, N, N//len(attr['trn_losses'])), attr['trn_losses'])
plt.plot(np.arange(0, N, N//len(attr['val_losses'])), attr['val_losses'])
plt.ylabel('Loss')
plt.xlabel('Batches')
plt.legend(['Training', 'Validation'])
plt.savefig(os.path.join(plotp, 'trn_progress'), bbox_inches='tight')
plt.close()