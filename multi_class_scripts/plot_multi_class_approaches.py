import numpy as np
import matplotlib.pyplot as plt
from ovseg.data.Dataset import raw_Dataset
from tqdm import tqdm
from ovseg.utils.label_utils import reduce_classes
from time import sleep
import os
from ovseg.utils.seg_fg_dial import seg_fg_dial

plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'workflows_multi_class')

if not os.path.exists(plotp):
    os.makedirs(plotp)

lb_classes = [1, 2, 9, 13, 15, 17]
ds = raw_Dataset('OV04')

for i in range(len(ds)):
    print(i)
    data_tpl = ds[i]
    lb = reduce_classes(data_tpl['label'], lb_classes)
    classes = np.unique(lb)
    if len(classes) < 4:
        continue
    print('found '+str(len(classes)-1)+' foreground classes')
# %%
    
    lb_oh = np.stack([lb == cl for cl in classes if cl > 0], 0).astype(float)
    
    lb_max_oh = np.max(lb_oh, (2, 3))
    
    lbs_z = np.sum(lb_max_oh, 0)
    
    if lbs_z.max() > 2:
        break

z = np.argmax(lbs_z)

# %%
plt.close()
plt.imshow(data_tpl['image'][z, 120:-80, 100:-100].clip(-150, 250), cmap='bone')

plt.axis('off')

plt.savefig(os.path.join(plotp, 'image'),bbox_inches='tight')

colors = ['red', 'blue', 'lime', 'orange', 'cyan', 'magenta']
classes_shown = []
ind = 0
for cl in classes:
    
    if cl == 0:
        continue
    
    seg = (lb[z, 120:-80, 100:-100] == cl).astype(float)
    
    if seg.max() > 0:
       plt.contour(seg, colors=colors[ind], linewidths=0.5) 
       ind += 1
       classes_shown.append(cl)
       
       plt.savefig(os.path.join(plotp, 'image_segs_{}'.format(ind)),bbox_inches='tight')

plt.close()
ind = 0
for cl in classes_shown:
    
    if cl == 0:
        continue
    plt.imshow(data_tpl['image'][z, 120:-80, 100:-100].clip(-150, 250), cmap='bone')
    plt.axis('off')
    
    seg = (lb[z, 120:-80, 100:-100] == cl).astype(float)
    
    plt.contour(seg, colors=colors[ind], linewidths=0.5) 
    ind += 1
       
    plt.savefig(os.path.join(plotp, 'image_class_{}'.format(int(cl))),bbox_inches='tight')
    plt.close()

# %% now the regions at risk

rar = seg_fg_dial(lb, 13, 5/0.66)

# %%
ind = 0
for cl in classes_shown:
    
    if cl == 0:
        continue
    plt.imshow(data_tpl['image'][z, 120:-80, 100:-100].clip(-150, 250), cmap='bone')
    plt.axis('off')
    
    seg = (rar[z, 120:-80, 100:-100] == cl).astype(float)
    
    plt.contour(seg, colors=colors[ind], linewidths=0.5) 
    ind += 1
       
plt.savefig(os.path.join(plotp, 'rar'.format(int(cl))),bbox_inches='tight')
plt.close()
# %%

plt.imshow(data_tpl['image'][z, 120:-80, 100:-100].clip(-150, 250), cmap='bone')
plt.axis('off')
for cl in classes_shown:
    seg = (lb[z, 120:-80, 100:-100] == cl).astype(float)
    
    plt.contour(seg, colors=colors[0], linewidths=0.5) 
plt.savefig(os.path.join(plotp, 'bin'.format(int(cl))),bbox_inches='tight')
plt.close()