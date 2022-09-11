import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import shutil
from tqdm import tqdm

plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'OV04',
                     'pod_om_4fCV', 'compare_heatmaps')

predp = os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'OV04',
                     'pod_om_4fCV')

a = 0.4
# %%
if os.path.exists(plotp):
    shutil.rmtree(plotp)

os.makedirs(plotp)

predp = os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'OV04',
                     'pod_om_4fCV')

# ds_list = ['BARTS', 'BARTS', 'ApolloTCGA', 'BARTS', 'BARTS']
# case_list = ['case_307', 'case_298', 'case_626', 'case_331', 'case_332']
# z_list = [44, 61, 53, 33, 79]
# bb_list = [[0, 512, 0, 512],
#            [0, 512, 0, 512],
#            [0, 512, 0, 512],
#            [0, 512, 0, 512],
#            [0, 512, 0, 512]]
# cl_list = [9,9,1,1,1]

n_plots = 20

d = 100
lw = 1

ds_list = n_plots*['ApolloTCGA']
case_list = [f'case_{c}' for c in range(600, 600 + n_plots)]
z_list = [44, 61,  79]
bb_list = [[100, 356, 128, 384],
           [128, 384, 168, 424],
           [50, 306, 128, 384]]
cl_list = n_plots//2*[9] + n_plots//2*[1]

# P = np.load(os.path.join(predp, 'P_cross_validation.npy'))
coefs = np.load(os.path.join(predp, 'coefs_v3.npy'))
for ds_name, case, cl in tqdm(zip(ds_list, case_list, cl_list)):
    
    imp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', ds_name, 'images')
    lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', ds_name, 'labels')
    
    
    gt = nib.load(os.path.join(lbp, f'{case}.nii.gz')).get_fdata()
    gt_cl = (gt == cl).astype(float)
    
    if gt_cl.max() == 0:
        continue
    
    z = np.argmax(np.sum(gt_cl, (0, 1)))
    X, Y = np.where(gt_cl[..., z])
    bb = [np.max([np.median(X) - d, 0]),
          np.min([np.median(X) + d, 512]),
          np.max([np.median(Y) - d, 0]),
          np.min([np.median(Y) + d, 512])]
    bb = np.array(bb).astype(int)
    
    im = nib.load(os.path.join(imp, f'{case}_0000.nii.gz')).get_fdata().clip(-50, 150)
    im = (im+50)/200

    hm_old = np.zeros_like(gt_cl)
    hm_drop = np.zeros_like(gt_cl)
    hm_new = np.zeros_like(gt_cl)
    
    # compute old heatmap
    for f in range(5,12):
        
        pred = nib.load(os.path.join(predp, 'calibrated_0.00', f'{ds_name}_fold_{f}',
                                       f'{case}.nii.gz')).get_fdata()
        
        hm_old += (pred == cl).astype(float)
    
    hm_old = hm_old/7
    
    # compute drop heatmap
    for f in range(7):
        
        pred = nib.load(os.path.join(predp, 'dropout_UNet_0', f'{ds_name}_{f}_fold_5', f'{case}.nii.gz')).get_fdata()
        
        hm_drop += (pred == cl).astype(float)
    
    hm_drop = hm_drop/7
    
    # compute new heatmap
    c = 0 if cl == 1 else 1
    a_w_list = coefs[c]#np.diff(np.concatenate([[0],P[:, c]]))
    
    for w, a_w in zip(range(-3,4), a_w_list):
        pred = nib.load(os.path.join(predp,
                                       f'calibrated_{w:.2f}',
                                       f'{ds_name}_ensemble_0_1_2_3',
                                       f'{case}.nii.gz')).get_fdata()
        hm_new += a_w * (pred == cl).astype(float)
    
    # final prediction
    pred = (hm_new > 0.5).astype(float)

    # images with segmentations
    plt.imshow(im[bb[0]:bb[1],bb[2]:bb[3], z]*(1-a), cmap='bone', vmax=1)
    plt.contour(gt_cl[bb[0]:bb[1],bb[2]:bb[3], z] > 0, colors='blue', linewidths=lw)
    # plt.contour(pred[bb[0]:bb[1],bb[2]:bb[3], z] > 0, colors='red', linewidths=lw)
    plt.axis('off')
    plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im1_image_seg.png'), bbox_inches='tight')
    plt.close()
    
    # # images with heatmap
    # plt.imshow(im[bb[0]:bb[1],bb[2]:bb[3], z], cmap='bone', alpha=1, vmax=1)
    # plt.imshow(hm_new[bb[0]:bb[1],bb[2]:bb[3], z], cmap='hot', alpha=a, vmax=1)
    # plt.axis('off')
    # plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im3_image_heat.png'), bbox_inches='tight')
    # plt.close()
    
    # just heatmap
    
    ext = ['unc', 'drop', 'old']
    for i, hm in enumerate([hm_new, hm_drop, hm_old]):
        
        plt.imshow(hm[bb[0]:bb[1],bb[2]:bb[3], z], cmap='hot', vmax=1)
        plt.contour(gt_cl[bb[0]:bb[1],bb[2]:bb[3], z] > 0, colors='blue', linewidths=lw)
        plt.axis('off')
        plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im{i+2}_{ext[i]}.png'), bbox_inches='tight')
        plt.close()

    
# %%


if False:
    
    from skimage.measure import label
    lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
                       'BARTS', 'labels')
    
    case_z_pairs = []
    
    for case in tqdm(os.listdir(lbp)):
        
        gt = nib.load(os.path.join(lbp, case)).get_fdata()
        lb = (gt==1).astype(float)
        
        for z in range(lb.shape[-1]):
            
            ccs, ncomps = label(lb[..., z], return_num=True)
            
            if ncomps > 4:
                case_z_pairs.append((case, z, ncomps))
# %%
predp = os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'kits21_trn',
                     'disease_3_1')

# ds_list = ['BARTS', 'BARTS', 'ApolloTCGA', 'BARTS', 'BARTS']
# case_list = ['case_307', 'case_298', 'case_626', 'case_331', 'case_332']
# z_list = [44, 61, 53, 33, 79]
# bb_list = [[0, 512, 0, 512],
#            [0, 512, 0, 512],
#            [0, 512, 0, 512],
#            [0, 512, 0, 512],
#            [0, 512, 0, 512]]
# cl_list = [9,9,1,1,1]

case_list = os.listdir(os.path.join(predp,'UQ_calibrated_0.00', 'kits21_tst_ensemble_0_1_2'))[0:10]

imp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'kits21', 'images')
lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'kits21', 'labels')

# P = np.load(os.path.join(predp, 'P_cross_validation.npy'))
coefs = np.load(os.path.join(predp, 'coefs_v3.npy'))
cl = 2
ds_name = 'kits21_tst'
for case in tqdm(case_list):
    
    case_id = case.split('.')[0]
    p = f'D:\\PhD\\kits21\\kits21\\data\\{case_id}\\segmentations'
    seg_files = [s for s in os.listdir(p) if s.startswith('tumor')]
    n_instances = len(seg_files) // 3
    
    
    for n in range(1,n_instances+1):
        segs = [nib.load(os.path.join(p,s)).get_fdata() for s in os.listdir(p) if s.startswith(f'tumor_instance-{n}')]
                
        im = nib.load(os.path.join(imp, case)).get_fdata().clip(-50, 150)
        im = (im+50)/200
        
        # get uncalibrated segmentations
            
        hm_new = np.zeros_like(segs[0])
        hm_drop = np.zeros_like(segs[0])
        hm_old = np.zeros_like(segs[0])
            
        # compute new heatmap
        a_w_list = coefs#np.diff(np.concatenate([[0],P[:, 0]]))
        
        for w, a_w in zip(range(-3,4), a_w_list):
            pred = nib.load(os.path.join(predp,
                                        f'UQ_calibrated_{w:.2f}',
                                        f'{ds_name}_ensemble_0_1_2',
                                        case)).get_fdata()
            hm_new += a_w * (pred == cl).astype(float)
        
        # compute old heatmap
        for f in range(3,10):
            
            pred = nib.load(os.path.join(predp, 'UQ_calibrated_0.00', f'{ds_name}_fold_{f}',
                                           case)).get_fdata()
            
            hm_old += (pred == cl).astype(float)
        
        hm_old = hm_old/7
    
    
        # compute drop heatmap
        for f in range(7):
            
            pred = nib.load(os.path.join(predp, 'dropout_UNet_0', f'{ds_name}_{f}_fold_3', case)).get_fdata()
            
            hm_drop += (pred == cl).astype(float)
        
        hm_drop = hm_drop/7
    
        z = np.argmax(np.sum(segs[0], (1,2)))
        X,Y = np.where(segs[0][z])
        x, y = int(np.median(X)), int(np.median(Y))
        bb = [np.max([0, x-d]), np.min([512, x+d]),
              np.max([0, y-d]), np.min([512, y+d])]
        
        # images with segmentations
        plt.imshow(im[z, bb[0]:bb[1],bb[2]:bb[3]], cmap='bone', vmax=1)
        hm_gt = np.mean(np.stack(segs, 0), 0)
        gt_cl = (hm_gt > 0.5).astype(float)
        # plt.imshow(hm_gt[z, bb[0]:bb[1],bb[2]:bb[3]], cmap='hot', alpha=a, vmax=1)
        plt.contour(gt_cl[z, bb[0]:bb[1],bb[2]:bb[3]] > 0, colors='blue', linewidths=lw)
        plt.axis('off')
        plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im2_image_seg.png'), bbox_inches='tight')
        plt.close()

        # heatmaps
        ext = ['unc', 'drop', 'old']
        for i, hm in enumerate([hm_new, hm_drop, hm_old]):
            
            plt.imshow(hm[z, bb[0]:bb[1],bb[2]:bb[3]], cmap='hot', vmax=1)
            plt.contour(gt_cl[z, bb[0]:bb[1],bb[2]:bb[3]] > 0, colors='blue', linewidths=lw)
            plt.axis('off')
            plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im{i+2}_{ext[i]}.png'), bbox_inches='tight')
            plt.close()
