import os
import pickle
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from ovseg.data.Dataset import raw_Dataset
from tqdm import tqdm
from rt_utils import RTStructBuilder

predbp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions',
                   'ApolloTCGA_BARTS_OV04','pod_om')

plotbp = os.path.join(os.environ['OV_DATA_BASE'], 'plots',
                   'ApolloTCGA_BARTS_OV04','ICON8_heatmaps')

if not os.path.exists(plotbp):
    os.makedirs(plotbp)
    
    
if not os.path.exists(os.path.join(predbp, 'heatmaps')):
    os.makedirs(os.path.join(predbp, 'heatmaps'))

# %% analyse the CV performance
mbp = os.path.join(os.environ['OV_DATA_BASE'], 'trained_models',
                   'ApolloTCGA_BARTS_OV04','pod_om')

all_models = [m for m in os.listdir(mbp) if m.startswith('cali')]

w1_list, w9_list = [], []
dsc1_list, dsc9_list = [], []

for model in all_models:
    params = pickle.load(open(os.path.join(mbp, model, 'model_parameters.pkl'), 'rb'))
    res = pickle.load(open(os.path.join(mbp, model, 'validation_CV_results.pkl'), 'rb'))
    w1_list.append(params['training']['loss_params']['loss_kwargs'][0]['w_list'][0])
    w9_list.append(params['training']['loss_params']['loss_kwargs'][0]['w_list'][1])
    dsc1_list.append(np.nanmean([res[scan]['dice_1'] for scan in res]))
    dsc9_list.append(np.nanmean([res[scan]['dice_9'] for scan in res]))

w_dsc1_sorted = np.array([(w, dsc) for w, dsc in sorted(zip(w1_list, dsc1_list))])
w_dsc9_sorted = np.array([(w, dsc) for w, dsc in sorted(zip(w9_list, dsc9_list))])

plt.subplot(1,2,1)
plt.plot(w_dsc1_sorted[:, 0], w_dsc1_sorted[:, 1])
plt.subplot(1,2,2)
plt.plot(w_dsc9_sorted[:, 0], w_dsc9_sorted[:, 1])

# compute weights with DSC better than best - 5
w1 = w_dsc1_sorted[w_dsc1_sorted[:, 1] >= w_dsc1_sorted[:, 1].max()-5, 0]
w9 = w_dsc9_sorted[w_dsc9_sorted[:, 1] >= w_dsc9_sorted[:, 1].max()-5, 0]

# we throw the last w from 9 out for symmetry
w9 = w9[:len(w1)]

models = [m for m in all_models if float(m.split('_')[1]) in w1 or float(m.split('_')[2]) in w9]

# %% plot heatmaps
ds = raw_Dataset('ICON8')

for data_tpl in tqdm(ds):
    om_hm = np.zeros_like(data_tpl['image'], dtype=float)
    pod_hm = np.zeros_like(data_tpl['image'], dtype=float)
    for m in models:
        predp = os.path.join(predbp, m, 'ICON8_ensemble_0_1_2_3_4')
        file = data_tpl['pat_name'] + '_' + data_tpl['date'] + '.nii.gz'
        pred = nib.load(os.path.join(predp, file)).get_fdata()
        
        if float(m.split('_')[1]) in w1:
            om_hm += (pred == 1).astype(float)
        if float(m.split('_')[2]) in w9:
            pod_hm += (pred == 9).astype(float)
    
    om_hm /= len(w1)
    pod_hm /= len(w9)
    
    for cl, seg_hm in zip(['om', 'pod'], [om_hm, pod_hm]):
        for z in np.where(np.sum(seg_hm, (1,2)))[0]:
            seg = seg_hm[z]
            im = (data_tpl['image'][z].clip(-150, 250) + 150)/400
            
            d = 50
            
            x, y = np.where(seg > 0)
            xmn, xmx, ymn, ymx = max([x.min()-d,0]), min([x.max()+d,512]), max([y.min()-d,0]), min([y.max()+d,512])
            
            plt.subplot(1,3,1)
            plt.imshow(im[xmn:xmx, ymn:ymx], 'gray')
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(np.stack([im + seg, im, im], -1)[xmn:xmx, ymn:ymx]/2)
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(seg[xmn:xmx, ymn:ymx], 'gray', vmax=1)
            plt.axis('off')
            plt.savefig(os.path.join(plotbp, data_tpl['pat_name']+'_'+cl+'_'+str(z)+'.png'))
            plt.close()

# %% save dcm_rt files
for data_tpl in tqdm(ds):
    
    dcm_path = data_tpl['raw_image_file']
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dcm_path)
    
    for m in models:
        predp = os.path.join(predbp, m, 'ICON8_ensemble_0_1_2_3_4')
        file = data_tpl['pat_name'] + '_' + data_tpl['date'] + '.nii.gz'
        pred = nib.load(os.path.join(predp, file)).get_fdata()
        pred = np.stack([pred[z] for z in range(pred.shape[0])], -1)
        w_om, w_pod = float(m.split('_')[1]), float(m.split('_')[2])                              
        
        for i, w in enumerate(w1):
            if w == w_om:
                mask = (pred == 1)   
                if mask.max() > 0:
                    rtstruct.add_roi(mask=mask[..., ::-1],
                                 color=[255, 0, 0],
                                 name=f'1-{i+1}',
                                 approximate_contours=False)
        for i, w in enumerate(w9):
            if w == w_pod:
                mask = (pred == 9)  
                if mask.max() > 0:              
                    rtstruct.add_roi(mask=mask[..., ::-1],
                                 color=[255, 0, 0],
                                 name=f'9-{i+1}',
                                 approximate_contours=False)
        
    rtstruct.save(os.path.join(predbp, 'heatmaps', data_tpl['pat_name']+'.dcm'))
