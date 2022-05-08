import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

# class cross_entropy(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.loss = torch.nn.CrossEntropyLoss(reduction='none')

#     def forward(self, logs, yb_oh, mask=None):
#         assert logs.shape == yb_oh.shape
#         yb_int = torch.argmax(yb_oh, 1)
#         l = self.loss(logs, yb_int)
#         if mask is not None:
#             l = l * mask[:, 0]
#         return l.mean()
class cross_entropy(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        yb_int = torch.argmax(yb_oh, 1)
        l = self.loss(logs, yb_int)
        if mask is not None:
            l = l * mask[:, 0]
        return l.mean()

class cross_entropy_exp_weight(nn.Module):

    def __init__(self, w_list):
        super().__init__()
        weight = torch.tensor([1] + [np.exp(-1*w) for w in w_list]).type(torch.float)
        if torch.cuda.is_available():
            weight = weight.cuda()
        self.loss = torch.nn.CrossEntropyLoss(weight=weight,
                                              reduction='none')

    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        yb_int = torch.argmax(yb_oh, 1)
        l = self.loss(logs, yb_int)
        if mask is not None:
            l = l * mask[:, 0]
        return l.mean()



class bin_cross_entropy(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        yb_int = torch.argmax(yb_oh, 1)
        yb_bin = (yb_int > 0).type(yb_int.dtype)
        logs_bin = torch.cat([logs[:, :1], logs[:, 1:].max(1, keepdim=True)[0]], 1)
        l = self.loss(logs_bin, yb_bin)
        if mask is not None:
            l = l * mask[:, 0]
        return l.mean()


class cross_entropy_weighted_bg(nn.Module):

    def __init__(self, weight_bg, n_fg_classes):
        super().__init__()
        self.weight_bg = weight_bg
        self.n_fg_classes = n_fg_classes
        self.weight = [self.weight_bg] + [1] * self.n_fg_classes
        self.weight = torch.tensor(self.weight).type(torch.float)
        if torch.cuda.is_available():
            self.weight = self.weight.cuda()
        self.loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='none')
        
    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        yb_int = torch.argmax(yb_oh, 1)
        l = self.loss(logs, yb_int)
        if mask is not None:
            l = l * mask[:, 0]
        return l.mean()

class cross_entropy_weighted_fg(nn.Module):

    def __init__(self, weights_fg):
        super().__init__()
        self.weights_fg = weights_fg
        assert isinstance(self.weights_fg, list), 'fg weights must be given as list'
        self.weight = [1] + self.weights_fg
        self.weight = torch.tensor(self.weight).type(torch.float)
        if torch.cuda.is_available():
            self.weight = self.weight.cuda()
        self.loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='none')
        
    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        yb_int = torch.argmax(yb_oh, 1)
        l = self.loss(logs, yb_int)
        if mask is not None:
            l = l * mask[:, 0]
        return l.mean()

class cross_entropy_weighted_fg_v2(nn.Module):

    def __init__(self, weights_fg):
        super().__init__()
        self.weights_fg = weights_fg
        assert isinstance(self.weights_fg, list), 'fg weights must be given as list'
        self.weight = [2 - np.mean(self.weights_fg)] + self.weights_fg
        self.weight = torch.tensor(self.weight).type(torch.float)
        if torch.cuda.is_available():
            self.weight = self.weight.cuda()
        self.loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='none')
        
    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        yb_int = torch.argmax(yb_oh, 1)
        l = self.loss(logs, yb_int)
        if mask is not None:
            l = l * mask[:, 0]
        return l.mean()


class dice_loss_weighted(nn.Module):

    def __init__(self, weight, eps=1e-5):
        # same as in the cross_entropy_weighted_bg: weight=1 means no weighting, weight < 1
        # mean more sens less precision
        super().__init__()
        self.eps = eps
        self.weight = weight
        self.w1 = (2-self.weight) * 0.5
        self.w2 = self.weight * 0.5

    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        pred = torch.nn.functional.softmax(logs, 1)
        # dimension in which we compute the mean
        dim = list(range(2, len(pred.shape)))
        # remove the background channel from both as the dice will only
        # be computed over foreground classes
        pred = pred[:, 1:]
        yb_oh = yb_oh[:, 1:]
        if mask is not None:
            pred = pred * mask
            # Is this second line neseccary? Probably not! But better be safe than sorry.
            yb_oh = yb_oh * mask
        # now compute the metrics
        tp = torch.sum(yb_oh * pred, dim)
        yb_vol = torch.sum(yb_oh, dim)
        pred_vol = torch.sum(pred, dim)
        # the main formula
        dice = (tp + self.eps) / (self.w1 * yb_vol + self.w2 * pred_vol + self.eps)
        # the mean is computed over the batch and channel axis (excluding background)
        return 1 - 1 * dice.mean()


class dice_loss(nn.Module):

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        pred = torch.nn.functional.softmax(logs, 1)
        # dimension in which we compute the mean
        dim = list(range(2, len(pred.shape)))
        # remove the background channel from both as the dice will only
        # be computed over foreground classes
        pred = pred[:, 1:]
        yb_oh = yb_oh[:, 1:]
        if mask is not None:
            pred = pred * mask
            # Is this second line neseccary? Probably not! But better be safe than sorry.
            yb_oh = yb_oh * mask
        # now compute the metrics
        tp = torch.sum(yb_oh * pred, dim)
        yb_vol = torch.sum(yb_oh, dim)
        pred_vol = torch.sum(pred, dim)
        # the main formula
        dice = (tp + self.eps) / (0.5 * yb_vol + 0.5 * pred_vol + self.eps)
        # the mean is computed over the batch and channel axis (excluding background)
        return 1 - 1 * dice.mean()

class dice_loss_sigm_weighted(nn.Module):

    def __init__(self, w_list, eps=1e-5):
        super().__init__()
        self.weights = torch.tensor([1 / (1 + np.exp(-1*w)) for w in w_list]).view(1, len(w_list))
        if torch.cuda.is_available():
            self.weights = self.weights.cuda().type(torch.float)
        self.eps = eps

    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        pred = torch.nn.functional.softmax(logs, 1)
        # dimension in which we compute the mean
        dim = list(range(2, len(pred.shape)))
        # remove the background channel from both as the dice will only
        # be computed over foreground classes
        pred = pred[:, 1:]
        yb_oh = yb_oh[:, 1:]
        if mask is not None:
            pred = pred * mask
            # Is this second line neseccary? Probably not! But better be safe than sorry.
            yb_oh = yb_oh * mask
        # now compute the metrics
        tp = torch.sum(yb_oh * pred, dim)
        fp = torch.sum((1 - yb_oh) * pred, dim)
        fn = torch.sum((1 - pred) * yb_oh, dim)
        # the main formula
        dice = (tp + self.eps) / (tp + self.weights * fp + (1-self.weights) * fn + self.eps)
        # the mean is computed over the batch and channel axis (excluding background)
        return 1 - 1 * dice.mean()

class bin_dice_loss(nn.Module):

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        pred = torch.nn.functional.softmax(logs, 1)
        # dimension in which we compute the mean
        dim = list(range(2, len(pred.shape)))
        # remove the background channel from both as the dice will only
        # be computed over foreground classes
        pred = pred[:, 1:].max(1, keepdim=True)[0]
        yb_oh = yb_oh[:, 1:].max(1, keepdim=True)[0]
        
        if mask is not None:
            pred = pred * mask
            # Is this second line neseccary? Probably not! But better be safe than sorry.
            yb_oh = yb_oh * mask
        # now compute the metrics
        tp = torch.sum(yb_oh * pred, dim)
        yb_vol = torch.sum(yb_oh, dim)
        pred_vol = torch.sum(pred, dim)
        # the main formula
        dice = (tp + self.eps) / (0.5 * yb_vol + 0.5 * pred_vol + self.eps)
        # the mean is computed over the batch and channel axis (excluding background)
        return 1 - 1 * dice.mean()


class dice_loss_vector_weighted(nn.Module):

    def __init__(self, weights, eps=1e-5):
        # same as in the cross_entropy_weighted_bg: weight=1 means no weighting, weight < 1
        # mean more sens less precision
        super().__init__()
        self.eps = eps
        assert isinstance(weights, list), 'weights must be given as a list'
        self.weights = torch.tensor(weights).type(torch.float).reshape((1, -1, 1))
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()
        
        self.w1 = self.weights * 0.5
        self.w2 = (2-self.weights) * 0.5

    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        
        pred = torch.nn.functional.softmax(logs, 1)
        
        # reshape to a 3d array, makes the multiplication with the weights
        # easier
        nb, nch = pred.shape[:2]
        pred = pred.reshape((nb, nch, -1))
        yb_oh = yb_oh.reshape((nb, nch, -1))
        
        # remove the background channel from both as the dice will only
        # be computed over foreground classes
        pred = pred[:, 1:]
        yb_oh = yb_oh[:, 1:]
        
        # apply loss mask if given.
        if mask is not None:
            pred = pred * mask
            # Is this second line neseccary? Probably not! But better be safe than sorry.
            yb_oh = yb_oh * mask

        # now compute overlap and volume
        tp = torch.sum(yb_oh * pred, 2)
        yb_vol = torch.sum(yb_oh, 2)
        pred_vol = torch.sum(pred, 2)
        # the main formula
        dice = (tp + self.eps) / (self.w1 * yb_vol + self.w2 * pred_vol + self.eps)
        # the mean is computed over the batch and channel axis (excluding background)
        return 1 - 1 * dice.mean()

class SLDS_loss(nn.Module):

    def __init__(self, weight_bg, n_fg_classes, weight_ds=1, weight_sl=1, eps=1e-5):
        super().__init__()

        self.weight_bg = weight_bg
        self.weight_ds = weight_ds
        self.weight_sl = weight_sl
        self.n_fg_classes = n_fg_classes
        self.eps = eps        

        self.ce_loss = cross_entropy()
        self.dice_loss = dice_loss(eps=self.eps)

        self.ce_loss_weighted = cross_entropy_weighted_bg(self.weight_bg, self.n_fg_classes-1)
        self.dice_loss_weighted = dice_loss_weighted(self.weight_bg)

    def forward(self, logs, yb_oh, mask=None):

        # for the segmentation of the large components we use the largest value in every channel 1 
        # as background and use channel 1 as foreground
        logs_sl_bg = torch.cat([logs[:, 0:], logs[:, 2:]], 1).max(1, keepdim=True)[0]
        logs_sl = torch.cat([logs_sl_bg, logs[:, 1:2]], 1)
        yb_oh_sl = torch.cat([yb_oh[:, 0:1] + yb_oh[:, 2:].sum(1, keepdim=True), yb_oh[:, 1:2]], 1)
        
        
        # for the detection of small lesions we use the first two channels as background and the
        # others as foreground
        logs_ds_bg = logs[:, :2].max(1, keepdim=True)[0]
        logs_ds = torch.cat([logs_ds_bg, logs[:, 2:]], 1)
        yb_oh_ds = torch.cat([yb_oh[:, :2].sum(1, keepdim=True), yb_oh[:, 2:]], 1)
        
        loss_sl = self.ce_loss(logs_sl, yb_oh_sl, mask) + self.dice_loss(logs_sl, yb_oh_sl, mask)
        loss_ds = self.ce_loss_weighted(logs_ds, yb_oh_ds, mask) + self.dice_loss_weighted(logs_ds, yb_oh_ds, mask)
        
        return self.weight_sl * loss_sl + self.weight_ds * loss_ds


class modifiedTverskyLoss(nn.Module):
    
    def __init__(self, gamma=0.5, delta=0.5, eps=1e-5):
        # with these constants this loss equals the DICE loss
        super().__init__()
        
        self.gamma = gamma
        self.delta = delta
        self.eps = eps
    
    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        
        pred = torch.nn.functional.softmax(logs, 1)
        
        # reshape to a 3d array, makes the multiplication with the weights
        # easier
        nb, nch = pred.shape[:2]
        pred = pred.reshape((nb, nch, -1))
        yb_oh = yb_oh.reshape((nb, nch, -1))
        
        # remove the background channel from both as the dice will only
        # be computed over foreground classes
        pred = pred[:, 1:]
        yb_oh = yb_oh[:, 1:]
        
        # apply loss mask if given.
        if mask is not None:
            mask = mask.reshape((nb, nch, -1))
            pred = pred * mask
            # Is this second line neseccary? Probably not! But better be safe than sorry.
            yb_oh = yb_oh * mask

        # now compute overlap and volume
        tp = torch.sum(yb_oh * pred, 2)
        fp = torch.sum((1 - yb_oh) * pred, 2)
        fn = torch.sum((1 - pred) * yb_oh, 2)
        # the main formula
        mTI = (tp + self.eps) / (tp + self.delta * fp + (1-self.delta) * fn + self.eps)
        # the mean is computed over the batch and channel axis (excluding background)
        
        mTI_loss = torch.pow(1 - mTI, 1 - self.gamma)
        
        return  mTI_loss.mean()

class modifiedFocalLoss(nn.Module):
    
    def __init__(self, gamma=0.0, delta=0.5, scale=2.0, eps=1e-5):
        # with these constants the modified focal loss should equal the 
        # cross entropy
        
        super().__init__()
        self.gamma = gamma
        self.delta = delta
        self.scale = scale
        self.eps = eps
    
    def forward(self, logs, yb_oh, mask=None):
        
        pred = torch.nn.functional.softmax(logs, 1).clamp(self.eps, 1-self.eps)
        log_pred = torch.nn.functional.log_softmax(logs, 1)
        
        weight_bg = torch.pow(1 - pred[:, :1], self.gamma) * (1-self.delta)
        
        focal_bg = -1*self.scale * weight_bg * yb_oh[:, :1] * log_pred[:, :1]
        focal_fg = -1*self.scale * self.delta * yb_oh[:, 1:] * log_pred[:, 1:]
        focal = torch.cat([focal_bg, focal_fg], 1)
        
        return focal.sum(dim=1).mean()

class unifiedFocalLoss(nn.Module):
    
    def __init__(self, gamma, delta, eps=1e-5, scale=2.0):
        super().__init__()
        
        self.gamma = gamma
        self.delta = delta
        self.eps = eps
        self.scale = scale
    
        self.tversky_loss = modifiedTverskyLoss(gamma=self.gamma,
                                                delta=self.delta,
                                                eps=self.eps)
        
        self.focal_loss = modifiedFocalLoss(gamma=self.gamma,
                                            delta=self.delta,
                                            scale=self.scale)
        
    def forward(self, logs, yb_oh, mask):
        
        return self.tversky_loss(logs, yb_oh, mask) + self.focal_loss(logs, yb_oh, mask)
