import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import shutil
from tqdm import tqdm

plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'OV04',
                     'pod_om_4fCV', 'new_heatmaps')

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


ds_list = ['BARTS', 'BARTS', 'BARTS']
case_list = ['case_307', 'case_298',  'case_332']
z_list = [44, 61,  79]
bb_list = [[100, 356, 128, 384],
           [128, 384, 168, 424],
           [50, 306, 128, 384]]
cl_list = [9,9,1]

P = np.load(os.path.join(predp, 'P_cross_validation.npy'))
for ds_name, case, z, bb, cl in tqdm(zip(ds_list, case_list, z_list, bb_list, cl_list)):
    
    imp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', ds_name, 'images')
    lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', ds_name, 'labels')
    
    
    gt = nib.load(os.path.join(lbp, f'{case}.nii.gz')).get_fdata()
    gt_cl = (gt == cl).astype(float)
    
    im = nib.load(os.path.join(imp, f'{case}_0000.nii.gz')).get_fdata().clip(-50, 150)
    im = (im+50)/200
    
    # get uncalibrated segmentations
    preds_unc = [nib.load(os.path.join(predp, 'calibrated_0.00', f'{ds_name}_fold_{f}',
                                       f'{case}.nii.gz')).get_fdata()
                 for f in range(5,12)]
        
    # get calibrated segmentations
    preds_cal = [nib.load(os.path.join(predp,
                                       f'calibrated_{w:.2f}',
                                       f'{ds_name}_ensemble_0_1_2_3',
                                       f'{case}.nii.gz')).get_fdata()
                 for w in range(-3,4)]        

    hm_old = np.zeros_like(gt_cl)
    hm_new = np.zeros_like(gt_cl)
    
    # compute old heatmap
    for pred in preds_unc:
        
        hm_old += (pred == cl).astype(float)
    
    hm_old = hm_old/7
    
    # compute new heatmap
    c = 0 if cl == 1 else 1
    a_w_list = np.diff(np.concatenate([[0],P[:, c]]))
    
    for pred, a_w in zip(preds_cal, a_w_list):
        
        hm_new += a_w * (pred == cl).astype(float)
    
    # final prediction
    pred = (hm_new > 0.5).astype(float)

    # now plot the images
    
    plt.imshow(im[bb[0]:bb[1],bb[2]:bb[3], z]*(1-a), cmap='bone', vmax=1)
    plt.axis('off')
    plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im1_image.png'), bbox_inches='tight')
    plt.close()

    # images with segmentations
    plt.imshow(im[bb[0]:bb[1],bb[2]:bb[3], z]*(1-a), cmap='bone', vmax=1)
    plt.contour(gt_cl[bb[0]:bb[1],bb[2]:bb[3], z] > 0, colors='blue', linewidths=0.5)
    # plt.contour(pred[bb[0]:bb[1],bb[2]:bb[3], z] > 0, colors='red', linewidths=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im2_image_seg.png'), bbox_inches='tight')
    plt.close()
    
    # images with heatmap
    plt.imshow(im[bb[0]:bb[1],bb[2]:bb[3], z], cmap='bone', alpha=1, vmax=1)
    plt.imshow(hm_new[bb[0]:bb[1],bb[2]:bb[3], z], cmap='hot', alpha=a, vmax=1)
    plt.axis('off')
    plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im3_image_heat.png'), bbox_inches='tight')
    plt.close()
    
    # just heatmap
    plt.imshow(hm_new[bb[0]:bb[1],bb[2]:bb[3], z], cmap='hot', vmax=1)
    plt.axis('off')
    plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im4_heat.png'), bbox_inches='tight')
    plt.close()


# %%

P = np.load(os.path.join(predp, 'P_cross_validation.npy'))
for c, cl in enumerate([1, 9]):
    p = np.concatenate([[0],P[:, c]])[::-1]

    plt.imshow(p[:, np.newaxis], cmap='hot', vmax=1)
    for i in range(len(p)):
        plt.text(0.8, i, f'{p[i]:.3f}', fontsize=13)
    
    plt.axis('off')
    plt.savefig(os.path.join(plotp, f'colorbar_{cl}.png'), bbox_inches='tight')
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

case_list = os.listdir(os.path.join(predp,'UQ_calibrated_0.00', 'kits21_tst_ensemble_0_1_2'))[:10]

imp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'kits21', 'images')
lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'kits21', 'labels')

P = np.load(os.path.join(predp, 'P_cross_validation.npy'))
cl = 2
ds_name = 'kits21_tst'
for case in tqdm(case_list):
    
    
    
    gt = nib.load(os.path.join(lbp, case)).get_fdata()
    gt_cl = (gt == cl).astype(float)
    print(gt.shape)
    
    im = nib.load(os.path.join(imp, case)).get_fdata().clip(-50, 150)
    im = (im+50)/200
    
    # get uncalibrated segmentations
        
    # get calibrated segmentations
    preds_cal = [nib.load(os.path.join(predp,
                                       f'UQ_calibrated_{w:.2f}',
                                       f'{ds_name}_ensemble_0_1_2',
                                       case)).get_fdata()
                 for w in range(-3,4)]        

    hm_new = np.zeros_like(gt_cl)
        
    # compute new heatmap
    a_w_list = np.diff(np.concatenate([[0],P[:, 0]]))
    
    for pred, a_w in zip(preds_cal, a_w_list):
        
        hm_new += a_w * (pred == cl).astype(float)

    z = np.argmax(np.sum(gt_cl, (0,1)))
    X,Y = np.where(gt_cl[..., z])
    x, y = int(np.median(X)), int(np.median(Y))
    bb = [np.max([0, x-128]), np.min([512, x+128]),
          np.max([0, y-128]), np.min([512, y+128])]

    # now plot the images
    plt.imshow(im[bb[0]:bb[1],bb[2]:bb[3], z]*(1-a), cmap='bone', vmax=1)
    plt.axis('off')
    plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im1_image.png'), bbox_inches='tight')
    plt.close()

    # images with segmentations
    plt.imshow(im[bb[0]:bb[1],bb[2]:bb[3], z]*(1-a), cmap='bone', vmax=1)
    plt.contour(gt_cl[bb[0]:bb[1],bb[2]:bb[3], z] > 0, colors='blue', linewidths=0.5)
    # plt.contour(pred[bb[0]:bb[1],bb[2]:bb[3], z] > 0, colors='red', linewidths=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im2_image_seg.png'), bbox_inches='tight')
    plt.close()
    
    # images with heatmap
    plt.imshow(im[bb[0]:bb[1],bb[2]:bb[3], z], cmap='bone', alpha=1, vmax=1)
    plt.imshow(hm_new[bb[0]:bb[1],bb[2]:bb[3], z], cmap='hot', alpha=a, vmax=1)
    plt.axis('off')
    plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im3_image_heat.png'), bbox_inches='tight')
    plt.close()
    
    # just heatmap
    plt.imshow(hm_new[bb[0]:bb[1],bb[2]:bb[3], z], cmap='hot', vmax=1)
    plt.axis('off')
    plt.savefig(os.path.join(plotp, f'{case}_{z}_{cl}_im4_heat.png'), bbox_inches='tight')
    plt.close()

# %%
cl = 2
p = np.concatenate([[0],P[:, 0]])[::-1]

plt.imshow(p[:, np.newaxis], cmap='hot', vmax=1)
for i in range(len(p)):
    plt.text(0.8, i, f'{p[i]:.3f}', fontsize=13)

plt.axis('off')
plt.savefig(os.path.join(plotp, f'colorbar_{cl}.png'), bbox_inches='tight')
plt.close()