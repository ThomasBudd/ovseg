import numpy as np
import torch

def _2d_morph_conv(seg_oh, selem=None):

    assert torch.is_tensor(seg_oh), 'input seg must be a torch tensor'
    
    assert len(seg_oh.shape) == 4, 'seg must be 4d'
    
    nz = seg_oh.shape[1]
    
    if selem is None:
        selem = torch.tensor([[0, 1/5, 0], [1/5, 1/5, 1/5], [0, 1/5, 0]])
    else:
        if isinstance(selem, np.ndarray):
            selem = torch.from_numpy(selem)
        
        selem = selem / selem.sum()
    
    if len(selem.shape) == 2:
        selem = selem.unsqueeze(0)
    
    selem = selem.to(seg_oh.device).type(torch.float)
    selem = torch.stack(nz * [selem])

    padding = (selem.shape[2]//2, selem.shape[3]//2 )

    return torch.nn.functional.conv2d(seg_oh.type(torch.float),
                                      selem,
                                      padding=padding,
                                      groups=nz)

def dial_2d(seg_oh, selem=None):
    
    seg_conv = _2d_morph_conv(seg_oh, selem)
    
    return (seg_conv > 0).type(seg_oh.dtype)

def eros_2d(seg_oh, selem=None):
    
    seg_conv = _2d_morph_conv(seg_oh, selem)
    
    return (seg_conv >= 1-1e-5).type(seg_oh.dtype)

def opening_2d(seg_oh, selem=None):
    
    return dial_2d(eros_2d(seg_oh, selem), selem)

def closing_2d(seg_oh, selem=None):
    
    return eros_2d(dial_2d(seg_oh, selem), selem)

def morph_cleaning(seg, selem=None):
    
    assert torch.is_tensor(seg)
    # to one hot encoding (in batch dimension)
    n_cl = int(seg.max())
    
    if n_cl == 0:
        return seg
    
    seg_oh = torch.stack([seg == cl for cl in range(1, n_cl+1)]).type(torch.float)
    
    # cleaning with opening and closing
    seg_oh_clean = opening_2d(closing_2d(seg_oh, selem), selem)
    
    # now back to labels
    seg_clean = torch.zeros_like(seg)
    for cl in range(n_cl):
        seg_clean += (cl + 1)*seg_oh_clean[cl]
    
    return seg_clean