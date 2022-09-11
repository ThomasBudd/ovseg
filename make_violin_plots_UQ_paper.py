import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm


# %% ovarian cases

measures_ovarian = {cl:{'Trainee FP':{}, 'Trainee FN':{}, 'Trainee TP':{},
                        'Trainee TN':{}, 'Ground truth': {}} for cl in [1, 9]}
new_nii_path = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\BARTS\\labels'
old_nii_path = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\BARTS_old\\labels'
predp = os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'OV04',
                     'pod_om_4fCV')

coefs = np.load(os.path.join(predp, 'coefs_v3.npy'))


for case in tqdm(os.listdir(new_nii_path)):
    lb_gt = nib.load(os.path.join(new_nii_path, case)).get_fdata()
    lb_trn = nib.load(os.path.join(old_nii_path, case)).get_fdata()
    preds_cal = [nib.load(os.path.join(predp,
                                       f'calibrated_{w:.2f}',
                                       'BARTS_ensemble_0_1_2_3',
                                       case)).get_fdata().astype(np.uint8)
                 for w in range(-3,4)]        

    for c, cl in enumerate([1, 9]):
        
        gt = (lb_gt == cl).astype(float)
        trn = (lb_trn == cl).astype(float)
        trn_FP = trn * (1-gt)
        trn_FN = (1-trn) * gt
        trn_TP = trn * gt
        trn_TN = (1-trn) * (1-gt)
        hm_new = np.zeros_like(gt)
        
        for pred, coef in zip(preds_cal, coefs[c, :]):
            
            hm_new += coef * (pred == cl).astype(float)
        
        # compute the metrics
        for mask, key in zip([trn_FP, trn_FN, trn_TP, trn_TN, gt],
                             ['Trainee FP', 'Trainee FN', 'Trainee TP', 'Trainee TN',
                              'Ground truth']):
            
            values, counts = np.unique(hm_new[mask == 1], return_counts=True)
            
            for value, count in zip(values, counts):
                
                if value not in measures_ovarian[cl][key]:
                    measures_ovarian[cl][key][value] = count
                else:
                    measures_ovarian[cl][key][value] += count

pickle.dump(measures_ovarian,
            open(os.path.join(predp, 'violin_data.pkl'), 'wb'))

# %%

# measures_ovarian = {2:{n: {} for n in [1,2,3]}}
# predp = os.path.join(os.environ['OV_DATA_BASE'],
#                      'predictions',
#                      'kits21_trn',
#                      'disease_3_1')

# coefs = np.load(os.path.join(predp, 'coefs_v3.npy'))
# case_list = os.listdir(os.path.join(predp,'UQ_calibrated_0.00', 'kits21_tst_ensemble_0_1_2'))

# imp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'kits21', 'images')
# lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'kits21', 'labels')

# cl = 2

# for case in tqdm(case_list):
    
#     case_id = case.split('.')[0]
#     p = f'D:\\PhD\\kits21\\kits21\\data\\{case_id}\\segmentations'
#     seg_files = [s for s in os.listdir(p) if s.startswith('tumor')]
#     n_instances = len(seg_files) // 3
    
#     im = nib.load(os.path.join(imp, case)).get_fdata().clip(-50, 150)
#     im = (im+50)/200   

#     hm_new = np.zeros_like(im)
        
#     for w, coef in zip(range(-3,4), coefs):
#         pred = nib.load(os.path.join(predp,
#                                        f'UQ_calibrated_{w:.2f}',
#                                        'kits21_tst_ensemble_0_1_2',
#                                        case)).get_fdata()
#         hm_new += coef * (pred == cl).astype(float)
    
#     for n in range(1,n_instances+1):
#         hm_trn = np.zeros_like(im)
        
#         for s in os.listdir(p):
#             if s.startswith(f'tumor_instance-{n}'):
#                 hm_trn += nib.load(os.path.join(p,s)).get_fdata()
        
#         for key in [1,2,3]:
            
#             mask = hm_trn == key
#             values, counts = np.unique(hm_new[mask], return_counts=True)
            
#             for value, count in zip(values, counts):
                
#                 if value not in measures_ovarian[cl][key]:
#                     measures_ovarian[cl][key][value] = count
#                 else:
#                     measures_ovarian[cl][key][value] += count
        
# pickle.dump(measures_ovarian,
#             open(os.path.join(predp, 'violin_data.pkl'), 'wb'))

# %%
plt.close()
predp = os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'OV04',
                     'pod_om_4fCV')
data_ovarian = pickle.load(open(os.path.join(predp, 'violin_data.pkl'), 'rb'))
predp = os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'kits21_trn',
                     'disease_3_1')
data_kits = pickle.load(open(os.path.join(predp, 'violin_data.pkl'), 'rb'))



data = data_ovarian
cl = 9


for c, cl in enumerate([9, 1]):
    i_list = ['Trainee FP', 'Trainee FN']#, 'Trainee TP', 'Trainee TN']
    
    violin_data = [[] for _ in range(len(i_list))]
    n = 10 ** 5
    for j, i in enumerate(i_list):
        
        n0 = np.sum([data[cl][i][key] for key in data[cl][i]])
        
        for key in data[cl][i]:
            if i == 'Trainee TN' and key == 0.0:
                continue
            
            violin_data[j].extend(int(data[cl][i][key] / 100 + 0.5) * [key])
    
    plt.subplot(1,2,c+1)
    plt.boxplot(violin_data)
    # plt.violinplot(violin_data)
    plt.xticks([1,2], ['Trainee FP', 'Trainee FN'])#, 'Trainee TP', 'Trainee TN'])
    plt.title('Pelvic/ovarian disease' if cl == 9 else 'Omental disease')

# %%
cl = 1

violin_data = [[] for _ in range(4)]
i_list = ['Trainee FP', 'Trainee FN', 'Trainee TP', 'Trainee TN']
n = 10 ** 5
for j, i in enumerate(i_list):
    
    n0 = np.sum([data[cl][i][key] for key in data[cl][i]])
    
    for key in data[cl][i]:
        if i == 'Trainee TN' and key == 0.0:
            continue
        violin_data[j].extend(int(data[cl][i][key] / 100 + 0.5) * [key])
    
print('Removing voxels with low prob.')
tr_list = np.linspace(0, 0.5, 51)
delta = np.zeros_like(tr_list)

for i, tr in enumerate(tr_list):
    
    delta[i] = np.sum(np.array(violin_data[0]) < tr) - np.sum(np.array(violin_data[2]) < tr)
    
print(delta)

print('Adding voxels with high prob.')
tr_list = np.linspace(0.5, 1.0, 51)
delta = np.zeros_like(tr_list)

for i, tr in enumerate(tr_list):
    
    delta[i] = np.sum(np.array(violin_data[1]) > tr) - np.sum(np.array(violin_data[3]) > tr)
    
print(delta)


