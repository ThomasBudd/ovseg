import os
import pydicom
import numpy as np
from ovseg.utils.io import save_dcmrt_from_data_tpl, read_dcms
from shutil import copytree
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import xnat
from ovseg.utils.torch_morph import morph_cleaning
import torch
from skimage.measure import label

np.random.seed(15031993)


tarp = 'D:\\PhD\\Data\\VTT'

for f in ['pelvic_ovarian', 'omentum', 'plots']:
    if not os.path.exists(os.path.join(tarp, f)):
        os.makedirs(os.path.join(tarp, f))

manual_path = 'D:\\PhD\\Data\\TCGA_new_RW_merged'
auto_path = 'D:\\PhD\\Data\\TCGA_new_TB_merged'

# %%

def remove_small_ccs(seg):
    
    mask = np.ones_like(seg)
    
    for z in range(seg.shape[0]):
        
        comps = label(seg[z])
        n_comps = comps.max()
        
        for i in range(1, n_comps+1):
            comp = comps == i
    
            if np.sum(comp.astype(float)) <= 10:
                mask[z][comp] = 0
        

    return seg * mask

# %% get CT sessions
xnathost   = 'https://vtt.medschl.cam.ac.uk/'
project_id = 'VisualTuringTest'

user_id    = 'vtt_owner1'
pwd        = 'Passw0rdVTT'

with xnat.connect(xnathost, user=user_id, password=pwd) as session:
    
    CT_sessions = []
    
    for experiment in session.experiments:
        
        xnat_experiment = session.experiments[experiment]
        
        if hasattr(xnat_experiment, 'data'):
            if 'modality' in xnat_experiment.data:
                
                if xnat_experiment.data['modality'] == 'CT':
                
                    CT_sessions.append(xnat_experiment.data['label'])
    
# %%

om_scans = []
pod_scans = []

for CT_session in CT_sessions:
    scan = CT_session[:12]
    task = CT_session[16:]
    
    if scan not in os.listdir(os.path.join(tarp, task)):
        if task == 'omentum':
            om_scans.append(scan)
        else:
            pod_scans.append(scan)


# %%

no_rois = []

# pod_scans = []
# om_scans = ['TCGA-61-2003']

print('copy omentum scans and chance rt file')
sleep(0.1)
for scan in tqdm(om_scans):
    copytree(os.path.join(auto_path, scan),
             os.path.join(tarp, 'omentum', scan))
    
    data_tpl = read_dcms(os.path.join(tarp, 'omentum', scan))
    
    seg = (data_tpl['label'] == 1).astype(float)
    
    seg = morph_cleaning(torch.from_numpy(seg).cuda()).cpu().numpy()
    
    data_tpl['label'] = remove_small_ccs(seg)
    
    if data_tpl['label'].max() == 0:
        print('WARNING: scan {} had no omentum'.format(scan))
        sleep(1)
        no_rois.append(os.path.join(tarp, 'omentum', scan))
    
    save_dcmrt_from_data_tpl(data_tpl,
                             out_file=data_tpl['raw_label_file'],
                             key='label',
                             names=['1'],
                             colors=[[255, 0, 0]])
    


print('copy omentum scans and chance rt file')
sleep(0.1)
for scan in tqdm(pod_scans):
    copytree(os.path.join(auto_path, scan),
             os.path.join(tarp, 'pelvic_ovarian', scan))
    data_tpl = read_dcms(os.path.join(tarp, 'pelvic_ovarian', scan))
    seg = (data_tpl['label'] == 9).astype(float)
    
    seg = morph_cleaning(torch.from_numpy(seg).cuda()).cpu().numpy()
    
    data_tpl['label'] = remove_small_ccs(seg)
    if data_tpl['label'].max() == 0:
        print('WARNING: scan {} had no POD'.format(scan))
    
        sleep(1)
        no_rois.append(os.path.join(tarp, 'pelvic_ovarian', scan))
    save_dcmrt_from_data_tpl(data_tpl,
                             out_file=data_tpl['raw_label_file'],
                             key='label',
                             names=['9'],
                             colors=[[0, 0, 255]])

if len(no_rois) > 0:
    print('No ROI found for scans')
    print(no_rois)

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
        