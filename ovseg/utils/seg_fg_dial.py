import numpy as np
import torch


def seg_fg_dial(seg, r, z_to_xy_ratio=1, use_3d_ops=False):
    assert isinstance(seg, np.ndarray), "Input must be nd array"

    if len(seg.shape) == 2:
        return seg_fg_dial_2d(seg, r)
    elif len(seg.shape) == 3:
        if use_3d_ops:
            return seg_fg_dial_3d(seg, r, z_to_xy_ratio)
        else:
            return seg_fg_dial_2d_stacked(seg, r)
    else:
        raise ValueError('Input shape must be 2d or 3d.')

def seg_eros(seg, r, z_to_xy_ratio=1, use_3d_ops=False):
    assert isinstance(seg, np.ndarray), "Input must be nd array"
    
    if len(seg.shape) == 2:
        return seg_eros_2d(seg, r)
    elif len(seg.shape) == 3:
        if use_3d_ops:
            return seg_eros_3d(seg, r, z_to_xy_ratio)
        else:
            return seg_eros_2d_stacked(seg, r)
    else:
        raise ValueError('Input shape must be 2d or 3d.')
    
    

# %%
def seg_eros_2d(seg, r):
    n_cl = int(seg.max())
    if n_cl == 0:
        return seg
    nx, ny = seg.shape
    
    # define the 2d circle used for the dialation
    circ = (np.sum(np.stack(np.meshgrid(np.linspace(-1, 1, 2*r+1), np.linspace(-1, 1, 2*r+1)))**2,0)<=1).astype(float)
    # seg to GPU and one hot encoding (excluding background)
    seg_gpu = torch.from_numpy(seg).cuda()
    seg_oh = torch.stack([seg_gpu==i for i in range(1, n_cl+1)], 0).unsqueeze(0).type(torch.float)
    # weight for convolution
    circ_gpu = torch.stack(n_cl*[torch.from_numpy(circ).cuda().unsqueeze(0)]).type(torch.float)
    
    # perform the convolution, the dialation can be obtained by thresholding
    seg_oh_conv = torch.nn.functional.conv2d(seg_oh, circ_gpu, padding=(r,r), groups=n_cl)
    seg_oh_eros = (seg_oh_conv[0] == 1).cpu().numpy()
    
    seg_eros = np.zeros((nx, ny))
    for c in range(n_cl):
        seg_eros[seg_oh_eros[c]] = c+1 
    return seg_eros

def seg_fg_dial_2d(seg, r):
    # performs the segmentation foreground dilation. All labels are dialated, if two original
    # ROIs are close original ROI will be kept and not overlapped by another class. If the
    # dialations overlap in voxel where previously there was foreground, the closest class
    # is assigned to this voxel
    
    # number of fg classes in the segmentation
    n_cl = int(seg.max())
    if n_cl == 0:
        return seg
    nx, ny = seg.shape
    
    # define the 2d circle used for the dialation
    circ = (np.sum(np.stack(np.meshgrid(np.linspace(-1, 1, 2*r+1), np.linspace(-1, 1, 2*r+1)))**2,0)<=1).astype(float)
    # seg to GPU and one hot encoding (excluding background)
    seg_gpu = torch.from_numpy(seg).cuda()
    seg_oh = torch.stack([seg_gpu==i for i in range(1, n_cl+1)], 0).unsqueeze(0).type(torch.float)
    # weight for convolution
    circ_gpu = torch.stack(n_cl*[torch.from_numpy(circ).cuda().unsqueeze(0)]).type(torch.float)
    
    # perform the convolution, the dialation can be obtained by thresholding
    seg_oh_conv = torch.nn.functional.conv2d(seg_oh, circ_gpu, padding=(r,r), groups=n_cl)
    seg_oh_dial = (seg_oh_conv[0] > 0).type(torch.float).cpu().numpy()
    
    # this dialation is done, but it is the "classical" one where original foreground can
    # be overlapped by the dialation of a closeby ROI
    seg_dial = np.argmax(np.concatenate([np.zeros((1, nx, ny)), seg_oh_dial], 0), 0)
    ovlp = (np.sum(seg_oh_dial, 0) > 1).astype(float)

    if ovlp.max() == 0:
        # if the dialated ROIs do not overlap we're done
        return seg_dial

    # else we have to do some more work
    # first compute the erosion edge of all ROIs
    cross = np.zeros((3,3))
    cross[1, :] = 1/5
    cross[:, 1] = 1/5
    
    # same trick: use GPU convolution instead of slow CPU version 
    # (we still have the seg_oh on the GPU)
    cross_gpu = torch.stack(n_cl*[torch.from_numpy(cross).cuda().unsqueeze(0)]).type(torch.float)
    seg_oh_conv = torch.nn.functional.conv2d(seg_oh, cross_gpu, padding=(1,1), groups=n_cl)
    seg_oh_eros_edge = (seg_oh - (seg_oh_conv[0] >= 1).type(torch.float)).cpu().numpy()[0]
    
    # step 1: voxel with overlap that were previously ROI and reset to this label
    seg_dial[seg > 0] = seg[seg > 0]
    ovlp[seg > 0] = 0
    
    # step 2: voxel with overlap that were previouly background are set to the closest class
    # this is a compute intensive part as we have to iterate over all voxel
    # coordinates
    coords_ovlp = np.stack(np.where(ovlp > 0), 1)

    # keeps the list of edge coordinates
    coords_eros_edge = [np.stack(np.where(seg_oh_eros_edge[i]),1) for i in range(n_cl)]
    
    for i, coord in enumerate(coords_ovlp):
    
        # compute the L2 distance to all edges for each class
        min_dists = [np.min(np.sum((coords - coords_ovlp[i:i+1])**2, 1))
                     for coords in coords_eros_edge]
        
        # +1 since background was excluded
        c = np.argmin(min_dists) + 1
        seg_dial[coord[0], coord[1]] = c

    return seg_dial

# %%
def seg_eros_2d_stacked(seg, r):
    # performs the segmentation foreground dilation for each slice. Expects the z axis
    # to be in first dimension
    # optimized for multiple slices
    
    # number of fg classes in the segmentation
    n_cl = int(seg.max())
    if n_cl == 0:
        return seg
    nz, nx, ny = seg.shape
    
    if nx != ny:
        print('Warning! nx != ny, expected tomographic image with z-axis first.')
    
    # define the 2d circle used for the dialation
    circ = (np.sum(np.stack(np.meshgrid(np.linspace(-1, 1, 2*r+1), np.linspace(-1, 1, 2*r+1)))**2,0)<=1).astype(float)
    circ /= circ.sum()
    # seg to GPU and one hot encoding (excluding background)
    seg_gpu = torch.from_numpy(seg).cuda()
    # now the z axis should be in the batch dimension and the classes are stacked in
    # the channel dimension
    seg_oh = torch.stack([seg_gpu==i for i in range(1, n_cl+1)], 1).type(torch.float)
    # weight for convolution
    circ_gpu = torch.stack(n_cl*[torch.from_numpy(circ).cuda().unsqueeze(0)]).type(torch.float)
    
    # perform the convolution, the dialation can be obtained by thresholding
    seg_oh_conv = torch.nn.functional.conv2d(seg_oh, circ_gpu, padding=(r,r), groups=n_cl)
    seg_oh_eros = (seg_oh_conv >= 1).cpu().numpy()
    
    seg_eros = np.zeros((nz, nx, ny))
    for c in range(n_cl):
        seg_eros[seg_oh_eros[:, c]] = c+1 
    return seg_eros


def seg_fg_dial_2d_stacked(seg, r):
    # performs the segmentation foreground dilation for each slice. Expects the z axis
    # to be in first dimension
    # optimized for multiple slices
    
    # number of fg classes in the segmentation
    n_cl = int(seg.max())
    if n_cl == 0:
        return seg
    nz, nx, ny = seg.shape
    
    if nx != ny:
        print('Warning! nx != ny, expected tomographic image with z-axis first.')
    
    # define the 2d circle used for the dialation
    circ = (np.sum(np.stack(np.meshgrid(np.linspace(-1, 1, 2*r+1), np.linspace(-1, 1, 2*r+1)))**2,0)<=1).astype(float)
    # seg to GPU and one hot encoding (excluding background)
    seg_gpu = torch.from_numpy(seg).cuda()
    # now the z axis should be in the batch dimension and the classes are stacked in
    # the channel dimension
    seg_oh = torch.stack([seg_gpu==i for i in range(1, n_cl+1)], 1).type(torch.float)
    # weight for convolution
    circ_gpu = torch.stack(n_cl*[torch.from_numpy(circ).cuda().unsqueeze(0)]).type(torch.float)
    
    # perform the convolution, the dialation can be obtained by thresholding
    seg_oh_conv = torch.nn.functional.conv2d(seg_oh, circ_gpu, padding=(r,r), groups=n_cl)
    seg_oh_dial = (seg_oh_conv > 0).type(torch.float).cpu().numpy()
    
    # this dialation is done, but it is the "classical" one where original foreground can
    # be overlapped by the dialation of a closeby ROI
    seg_dial = np.argmax(np.concatenate([np.zeros((nz, 1, nx, ny)), seg_oh_dial], 1), 1)
    ovlp = (np.sum(seg_oh_dial, 1) > 1).astype(float)

    if ovlp.max() == 0:
        # if the dialated ROIs do not overlap we're done
        return seg_dial

    # else we have to do some more work
    # first compute the erosion edge of all ROIs
    cross = np.zeros((3,3))
    cross[1, :] = 1/5
    cross[:, 1] = 1/5
    
    # same trick: use GPU convolution instead of slow CPU version 
    # (we still have the seg_oh on the GPU)
    cross_gpu = torch.stack(n_cl*[torch.from_numpy(cross).cuda().unsqueeze(0)]).type(torch.float)
    seg_oh_conv = torch.nn.functional.conv2d(seg_oh, cross_gpu, padding=(1,1), groups=n_cl)
    seg_oh_eros_edge = (seg_oh - (seg_oh_conv >= 1).type(torch.float)).cpu().numpy()
    
    # step 1: voxel with overlap that were previously ROI and reset to this label
    seg_dial[seg > 0] = seg[seg > 0]
    ovlp[seg > 0] = 0
    
    # step 2: voxel with overlap that were previouly background are set to the closest class
    # this is a compute intensive part as we have to iterate over all voxel
    # coordinates
    for z in range(nz):
        coords_ovlp = np.stack(np.where(ovlp[z] > 0), 1)
    
        # keeps the list of edge coordinates
        coords_eros_edge = [np.stack(np.where(seg_oh_eros_edge[z, i]),1) for i in range(n_cl)]
        
        for i, coord in enumerate(coords_ovlp):
        
            
            # compute the L2 distance to all edges for each class
            min_dists = []
            for coords in coords_eros_edge:
                if len(coords) > 0:
                    min_dists.append(np.min(np.sum((coords - coords_ovlp[i:i+1])**2, 1)))
                else:
                    min_dists.append(np.inf)

            # +1 since background was excluded
            c = np.argmin(min_dists) + 1
            seg_dial[z, coord[0], coord[1]] = c

    return seg_dial

# %%
def seg_eros_3d(seg, r, z_to_xy_ratio):
    # full 3d version where 3d operations are used for the dialation
    n_cl = int(seg.max())
    if n_cl == 0:
        return seg
    nz, nx, ny = seg.shape
    
    if nx != ny:
        print('Warning! nx != ny, expected tomographic image with z-axis first.')
    
    # define the 2d circle used for the dialation
    rz = int(r/z_to_xy_ratio + 0.5)
    circ = (np.sum(np.stack(np.meshgrid(*[np.linspace(-1, 1, 2*R+1) for R in [rz, r, r]], indexing='ij'))**2,0)<=1).astype(float)
    circ /= circ.sum()
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
    seg_oh_eros = (seg_oh_conv[0] >= 1).cpu().numpy()
    
    seg_eros = np.zeros((nz, nx, ny))
    for c in range(n_cl):
        seg_eros[seg_oh_eros[c]] = c+1 
    return seg_eros


def seg_fg_dial_3d(seg, r, z_to_xy_ratio):
    # full 3d version where 3d operations are used for the dialation
    n_cl = int(seg.max())
    if n_cl == 0:
        return seg
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
        return seg_dial

    
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
    
    return seg_dial