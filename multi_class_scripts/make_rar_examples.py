import numpy as np
import torch
import os
from ovseg.utils.label_utils import reduce_classes
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

r = 13
z_to_xy_ratio = 8
lb_classes = [1,9,2,13,15,17]

lbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
                   'OV04', 'labels')
imp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
                   'OV04', 'images')

def seg_fg_dial_3d(seg):
    # full 3d version where 3d operations are used for the dialation
    n_cl = int(seg.max())
    if n_cl == 0:
        return seg, []
    nz, nx, ny = seg.shape
    
    if nx != ny:
        print('Warning! nx != ny, expected tomographic image with z-axis first.')
    
    # define the 2d circle used for the dialation
    rz = int(r/z_to_xy_ratio + 0.5)
    circ = (np.sum(np.stack(np.meshgrid(*[np.linspace(-1, 1, 2*R+1) for R in [rz, r, r]], indexing='ij'))**2,0)<=1).astype(float)
    # seg to GPU and one hot encoding (excluding background)
    seg_gpu = torch.from_numpy(seg).cuda()
    seg_gpu.requires_grad = False
    # now the z axis should be in the batch dimension and the classes are stacked in
    # the channel dimension
    seg_oh = torch.stack([seg_gpu==i for i in range(1, n_cl+1)], 0).type(torch.float).unsqueeze(0)
    # weight for convolution
    circ_gpu = torch.stack(n_cl*[torch.from_numpy(circ).cuda().unsqueeze(0)]).type(torch.float)
    
    # perform the convolution, the dialation can be obtained by thresholding
    seg_oh_conv = torch.nn.functional.conv3d(seg_oh, circ_gpu, padding=(rz,r,r), groups=n_cl)
    seg_oh_dial = (seg_oh_conv.cpu().numpy() > 0).astype(float)
    
    # this dialation is done, but it is the "classical" one where original foreground can
    # be overlapped by the dialation of a closeby ROI
    seg_dial = np.argmax(np.concatenate([np.zeros((1, nz, nx, ny)), seg_oh_dial[0]], 0), 0)
    ovlp = (np.sum(seg_oh_dial[0], 0) > 1).astype(float)
    
    if ovlp.max() == 0:
        # if the dialated ROIs do not overlap we're done
        return seg_dial, []

    
    print('overlap detected')
    # else we have to do some more work
    # first compute the erosion edge of all ROIs
    cross = np.zeros((3, 3, 3))
    cross[1, :, 1] = 1/7
    cross[:, 1, 1] = 1/7
    cross[1, 1, :] = 1/7
    
    # same trick: use GPU convolution instead of slow CPU version 
    # (we still have the seg_oh on the GPU)
    cross_gpu = torch.stack(n_cl*[torch.from_numpy(cross).cuda().unsqueeze(0)]).type(torch.float)
    seg_oh_conv = torch.nn.functional.conv3d(seg_oh, cross_gpu, padding=(1,1,1), groups=n_cl)
    seg_oh_eros_edge = (seg_oh - (seg_oh_conv >= 1).type(torch.float)).cpu().numpy()[0]
    
    # step 1: voxel with overlap that were previously ROI and reset to this label
    seg_dial[seg > 0] = seg[seg > 0]
    ovlp[seg > 0] = 0
    
    # step 2: voxel with overlap that were previouly background are set to the closest class
    # this is a compute intensive part as we have to iterate over all voxel
    # coordinates
    coords_ovlp = np.stack(np.where(ovlp > 0), 1)
    
    # keeps the list of edge coordinates
    coords_eros_edge = [np.stack(np.where(seg_oh_eros_edge[i]),1) for i in range(n_cl)]
    
    w = np.array([z_to_xy_ratio, 1, 1]).reshape((1,3))
    
    for i, coord in enumerate(coords_ovlp):
    
        
        # compute the L2 distance to all edges for each class
        min_dists = []
        for coords in coords_eros_edge:
            if len(coords) > 0:
                min_dists.append(np.min(np.sum(w * (coords - coords_ovlp[i:i+1])**2, 1)))
            else:
                min_dists.append(np.inf)
    
        # +1 since background was excluded
        c = np.argmin(min_dists) + 1
        seg_dial[coord[0], coord[1], coord[2]] = c
    
    return seg_dial, np.where(np.sum(ovlp, (1,2)))[0]

# %%
for k, scan in enumerate(tqdm(os.listdir(lbp))):
    
    seg = nib.load(os.path.join(lbp, scan)).get_fdata()
    if seg.max() == 0:
        continue
    seg = np.moveaxis(seg, -1, 0)
    seg = reduce_classes(seg, [1, 9])
    seg_dial, z_list = seg_fg_dial_3d(seg)
    


    # %%
    im = nib.load(os.path.join(imp, scan.split('.')[0]+'_0000.nii.gz')).get_fdata()
    if len(z_list) > 0:
       z = np.median(z_list).astype(int)
    else:
        z = np.argmax(np.sum(seg > 0, (1, 2)))

    # %%
    a, ad = 0.5, 0.0
    s1, s2 = (seg[z] == 1).astype(float), (seg[z] == 2).astype(float)
    sd1, sd2 = (seg_dial[z] == 1).astype(float), (seg_dial[z] == 2).astype(float)
    i = (im[..., z].clip(-150, 250) + 150)/400
    
    ovl = np.stack([i + s1*a + sd1*ad, i, i + s2*a + sd2*ad], -1) / (1+a+ad)
    
    plt.imshow(ovl)
    plt.contour(seg[z] == 1, colors='red')
    plt.contour(seg[z] == 2, colors='blue')
    plt.contour(seg_dial[z] == 1, colors='red', alpha=0.5, linestyles='dotted')
    plt.contour(seg_dial[z] == 2, colors='blue', alpha=0.5, linestyles='dotted')
    plt.axis('off')
    plt.savefig(os.path.join(os.environ['OV_DATA_BASE'],
                             'plots',
                             'OV04',
                             'regions_{}.png'.format(k+1)))
    plt.close()
    print(k)

    if k == 5:
        break
