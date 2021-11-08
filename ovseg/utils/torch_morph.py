import numpy as np
import torch

def _2d_morph_conv(seg_oh):

    assert torch.is_tensor(seg_oh), 'input seg must be a torch tensor'
    
    assert len(seg_oh.shape) == 4, 'seg must be 4d'
    
    nz = seg_oh.shape[1]
    selem = torch.tensor([[0, 1/5, 0], [1/5, 1/5, 1/5], [0, 1/5, 0]]).view((1,3,3))
    selem = selem.to(seg_oh.device)
    selem = torch.stack(nz * [selem])

    return torch.nn.functional.conv2d(seg_oh,
                                      selem,
                                      padding=(1,1),
                                      groups=nz)

def _2d_dial(seg_oh):
    
    seg_conv = _2d_morph_conv(seg_oh)
    
    return (seg_conv > 0).type(seg_oh.dtype)

def _2d_eros(seg_oh):
    
    seg_conv = _2d_morph_conv(seg_oh)
    
    return (seg_conv >= 1-1e-5).type(seg_oh.dtype)

def _2d_opening(seg_oh):
    
    return _2d_dial(_2d_eros(seg_oh))

def _2d_closing(seg_oh):
    
    return _2d_eros(_2d_dial(seg_oh))

def morph_cleaning(seg):
    
    assert torch.is_tensor(seg)
    # to one hot encoding (in batch dimension)
    n_cl = int(seg.max())
    
    if n_cl == 0:
        return seg
    
    seg_oh = torch.stack([seg == cl for cl in range(1, n_cl+1)]).type(torch.float)
    
    # cleaning with opening and closing
    seg_oh_clean = _2d_opening(_2d_closing(seg_oh))
    
    # now back to labels
    seg_clean = torch.zeros_like(seg)
    for cl in range(n_cl):
        seg_clean += (cl + 1)*seg_oh_clean[cl]
    
    return seg_clean