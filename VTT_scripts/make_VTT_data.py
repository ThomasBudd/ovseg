import os
import pydicom
import numpy as np
from ovseg.utils.io import save_dcmrt_from_data_tpl, read_dcms
from shutil import copytree
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt

np.random.seed(15031993)


tarp = 'D:\\PhD\\Data\\VTT'

for f in ['pelvic_ovarian', 'omentum', 'plots']:
    if not os.path.exists(os.path.join(tarp, f)):
        os.makedirs(os.path.join(tarp, f))

manual_path = 'D:\\PhD\\Data\\TCGA_new_RW_merged'
auto_path = 'D:\\PhD\\Data\\TCGA_new_TB_merged'

for path in [manual_path, auto_path]:
    
    scans = os.listdir(path)
    scans = np.random.choice(scans, size=27, replace=False)
    
    has_1 = []
    has_9 = []
    print('Find files with 1 or 9 from ', path)
    sleep(0.1)
    for scan in tqdm(os.listdir(path)):
        
        data_tpl = read_dcms(os.path.join(path, scan))
        
        if 'label' not in data_tpl:
            continue
        
        lb = data_tpl['label']
        
        if (lb == 1).max() > 0:
            has_1.append(scan)
        if (lb == 9).max() > 0:
            has_9.append(scan)
    
    # %%
    om_scans = np.random.choice(has_1, size=20, replace=False)
    pod_scans = np.random.choice(has_9, size=20, replace=False)
    
    # this was a hack to repair the fact that 25-1634 was broken
    # om_scans = ['TCGA-25-1635']
    # pod_scans = []
    
    print('copy omentum scans and chance rt file')
    sleep(0.1)
    for scan in tqdm(om_scans):
        copytree(os.path.join(path, scan),
                 os.path.join(tarp, 'omentum', scan))
        
        data_tpl = read_dcms(os.path.join(tarp, 'omentum', scan))
        data_tpl['label'] = (data_tpl['label'] == 1).astype(float)
        save_dcmrt_from_data_tpl(data_tpl,
                                 out_file=data_tpl['raw_label_file'],
                                 key='label',
                                 names=['1'],
                                 colors=[[255, 0, 0]])
    print('copy omentum scans and chance rt file')
    sleep(0.1)
    for scan in tqdm(pod_scans):
        copytree(os.path.join(path, scan),
                 os.path.join(tarp, 'pelvic_ovarian', scan))
        data_tpl = read_dcms(os.path.join(tarp, 'pelvic_ovarian', scan))
        data_tpl['label'] = (data_tpl['label'] == 9).astype(float)
        save_dcmrt_from_data_tpl(data_tpl,
                                 out_file=data_tpl['raw_label_file'],
                                 key='label',
                                 names=['9'],
                                 colors=[[0, 0, 255]])

# %%

for fol, ext in zip(['pelvic_ovarian', 'omentum'], ['_pod', '_om']):
    for scan in tqdm(os.listdir(os.path.join(tarp, fol))):
        data_tpl = read_dcms(os.path.join(tarp, fol, scan))
        if data_tpl['spacing'][0] < 3:
            print(data_tpl['spacing'][0], scan)
        
        im = data_tpl['image'].clip(-150, 250)
        lb = data_tpl['label']
        z_list = np.where(np.sum(lb, (1,2)))[0]
        for z in z_list:
            plt.imshow(im[z], cmap='bone')
            plt.contour(lb[z])
            plt.axis('off')
            plt.savefig(os.path.join(tarp, 'plots', scan + ext + '_'+str(z)),
                        bbox_inches='tight')
            plt.close()
        