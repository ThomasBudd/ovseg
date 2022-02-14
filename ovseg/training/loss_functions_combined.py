import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ovseg.training.loss_functions import cross_entropy, dice_loss
from ovseg.training.loss_functions import __dict__ as loss_functions_dict

def to_one_hot_encoding(yb, n_ch):

    yb = yb.long()
    yb_oh = torch.cat([(yb == c) for c in range(n_ch)], 1).float()
    return yb_oh


class CE_dice_loss(nn.Module):
    # weighted sum of the two losses
    # this functions is just here for historic reason
    def __init__(self, eps=1e-5, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_loss = cross_entropy()
        self.dice_loss = dice_loss(eps)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logs, yb, mask=None):
        if yb.shape[1] == 1:
            # turn yb to one hot encoding
            yb = to_one_hot_encoding(yb, logs.shape[1])
        ce = self.ce_loss(logs, yb, mask) * self.ce_weight
        dice = self.dice_loss(logs, yb, mask) * self.dice_weight
        loss = ce + dice
        return loss

class weighted_combined_loss(nn.Module):
    # arbritray loss functions weighted and summed up
    def __init__(self, loss_names, loss_weights=None, loss_kwargs=None):
        super().__init__()
        self.loss_names = loss_names
        # if no weights are given the losses are just summed without any weight
        self.loss_weights = loss_weights if loss_weights is not None else [1] * len(self.loss_names)
        # if no kwargs are given we just use blanks
        
        if loss_kwargs is None:
            
            self.loss_kwargs = [{}] * len(self.loss_names)
        
        elif len(loss_names) == 1 and isinstance(loss_kwargs, dict):
            self.loss_kwargs = [loss_kwargs]
        else:
            self.loss_kwargs = loss_kwargs
            

        assert len(loss_names) > 0, 'no names for losses given.'
        assert len(loss_names) == len(self.loss_weights), 'Got different amount of loss names and weights'
        assert len(loss_names) == len(self.loss_kwargs), 'Got different amount of loss names and kwargs'
        
        
        self.losses = []        
        for name, kwargs in zip(self.loss_names, self.loss_kwargs):
            if name not in loss_functions_dict:
                losses_found = [key for key in loss_functions_dict
                                if not key.startswith('_') and key not in ['torch', 'nn', 'np']]
                raise ValueError('Name {} for a loss functions was not found in loss_functions.py. '
                                 ' Got the modules {}'.format(name, losses_found))
            self.losses.append(loss_functions_dict[name](**kwargs))

        self.losses = nn.ModuleList(self.losses)

    def forward(self, logs, yb, mask=None):
        if yb.shape[1] == 1:
            # turn yb to one hot encoding
            yb = to_one_hot_encoding(yb, logs.shape[1])
            
        l = self.losses[0](logs, yb, mask) * self.loss_weights[0]
        for loss, weight in zip(self.losses[1:], self.loss_weights[1:]):
            l += loss(logs, yb, mask) * weight

        return l

def downsample_yb(logs_list, yb):
    
    # get pytorch 2d or 3d adaptive max pooling function
    f = F.adaptive_max_pool3d if len(yb.shape) == 5 else F.adaptive_max_pool2d
    
    # target downsampled to same size as logits
    return [f(yb, logs.shape[2:]) for logs in logs_list]
    

def downsample_yb_old(logs_list, yb):
    # NOT IN USAGE ANYMORE
    # ugly implementation of maxpooling, replaced by function 'downsample_yb'
    
    # this function downsamples the target (or masks) to the same shapes as the outputs
    # from the different resolutions of the decoder path of the U-Net.
    yb_list = [yb]
    is_3d = len(yb.shape) == 5
    for logs in logs_list[1:]:
        if is_3d:
            # maybe downsample in first spatial direction
            if logs.shape[2] == yb.shape[2] // 2:
                yb = torch.maximum(yb[:, :, ::2], yb[:, :, 1::2])
            elif not logs.shape[2] == yb.shape[2]:
                raise ValueError('shapes of logs and labels aren\'t machting for '
                                 'downsampling. got {} and {}'
                                 .format(logs.shape, yb.shape))
            # maybe downsample in second spatial direction
            if logs.shape[3] == yb.shape[3] // 2:
                yb = torch.maximum(yb[:, :, :, ::2], yb[:, :, :, 1::2])
            elif not logs.shape[3] == yb.shape[3]:
                raise ValueError('shapes of logs and labels aren\'t machting for '
                                 'downsampling. got {} and {}'
                                 .format(logs.shape, yb.shape))
            # maybe downsample in third direction
            if logs.shape[4] == yb.shape[4] // 2:
                yb = torch.maximum(yb[:, :, :, :, ::2], yb[:, :, :, :, 1::2])
            elif not logs.shape[4] == yb.shape[4]:
                raise ValueError('shapes of logs and labels aren\'t machting for '
                                 'downsampling. got {} and {}'
                                 .format(logs.shape, yb.shape))
        else:
            # maybe downsample in first spatial direction
            if logs.shape[2] == yb.shape[2] // 2:
                yb = yb[:, :, ::2] + yb[:, :, 1::2]
            elif not logs.shape[2] == yb.shape[2]:
                raise ValueError('shapes of logs and labels aren\'t machting for '
                                 'downsampling. got {} and {}'
                                 .format(logs.shape, yb.shape))
            # maybe downsample in second spatial direction
            if logs.shape[3] == yb.shape[3] // 2:
                yb = yb[:, :, :, ::2] + yb[:, :, :, 1::2]
            elif not logs.shape[3] == yb.shape[3]:
                raise ValueError('shapes of logs and labels aren\'t machting for '
                                 'downsampling. got {} and {}'
                                 .format(logs.shape, yb.shape))
            # now append
        yb_list.append(yb)
    return yb_list


class CE_dice_pyramid_loss(nn.Module):

    def __init__(self, eps=1e-5, ce_weight=1.0, dice_weight=1.0,
                 pyramid_weight=0.5):
        super().__init__()
        self.ce_dice_loss = CE_dice_loss(eps, ce_weight, dice_weight)
        self.pyramid_weight = pyramid_weight

    def forward(self, logs_list, yb, mask=None):
        if yb.shape[1] == 1:
            yb = to_one_hot_encoding(yb, logs_list[0].shape[1])
        # compute the weights to be powers of pyramid_weight
        scale_weights = self.pyramid_weight ** np.arange(len(logs_list))
        # let them sum to one
        scale_weights = scale_weights / np.sum(scale_weights)
        # turn labels into one hot encoding and downsample to same resolutions
        # as the logits
        yb_list = downsample_yb(logs_list, yb)
        if torch.is_tensor(mask):
            mask_list = downsample_yb(logs_list, mask)
        else:
            mask_list = [None] * len(yb_list)

        # now let's compute the loss for each scale
        loss = 0
        for logs, yb, m, w in zip(logs_list, yb_list, mask_list, scale_weights):
            loss += w * self.ce_dice_loss(logs, yb, m)

        return loss

class weighted_combined_pyramid_loss(nn.Module):

    def __init__(self, loss_names, loss_weights=None, loss_kwargs=None, pyramid_weight=0.5):
        super().__init__()
        self.loss = weighted_combined_loss(loss_names, loss_weights, loss_kwargs)
        self.pyramid_weight = pyramid_weight

  
    def forward(self, logs_list, yb, mask=None):
        if yb.shape[1] == 1:
            yb = to_one_hot_encoding(yb, logs_list[0].shape[1])
        # compute the weights to be powers of pyramid_weight
        scale_weights = self.pyramid_weight ** np.arange(len(logs_list))
        # let them sum to one
        scale_weights = scale_weights / np.sum(scale_weights)
        # turn labels into one hot encoding and downsample to same resolutions
        # as the logits
        yb_list = downsample_yb(logs_list, yb)
        if torch.is_tensor(mask):
            mask_list = downsample_yb(logs_list, mask)
        else:
            mask_list = [None] * len(yb_list)

        # now let's compute the loss for each scale
        loss = 0
        for logs, yb, m, w in zip(logs_list, yb_list, mask_list, scale_weights):
            loss += w * self.loss(logs, yb, m)
        return loss