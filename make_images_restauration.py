import numpy as np
from os import environ
from os.path import join
import matplotlib.pyplot as plt

prep = join(environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_2d')

case = 'case_300.npy'

fols = ['images'] + ['restaurations_'+ext for ext in ['full', 'half', 'quater', 'eights', '16']]

t_list = ['ground truth', 'full', 'half', 'quater', 'eights', '16']

lb = np.load(join(prep, 'labels', case))
z = np.argmax(np.sum(lb, (1, 2)))

for i, fol in enumerate(fols):
    plt.subplot(2, 3, i+1)
    im = np.load(join(prep, fol, case)).astype(float)
    plt.imshow(im[z], cmap='bone')
    plt.axis('off')
    plt.title(t_list[i])


#%%
x, y = np.where(lb[z] > 0)
x1, x2 = x.min(), x.max()
y1, y2 = y.min(), y.max()
plt.figure()

for i, fol in enumerate(fols):
    plt.subplot(2, 3, i+1)
    im = np.load(join(prep, fol, case)).astype(float)
    plt.imshow(im[z, x1-10:x2+11, y1-10:y2+11], cmap='bone')
    if i == 0:
        plt.contour(lb[z, x1-10:x2+11, y1-10:y2+11], colors='red')
    plt.axis('off')
    plt.title(t_list[i])
