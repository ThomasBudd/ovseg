import torch
from torch import nn
import numpy as np


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
        logs_sl_bg = torch.cat([logs[:, 0:], logs[:, 2:]]).max(1, keepdim=True)[0]
        logs_sl = torch.cat([logs_sl_bg, logs[:, 1:2]], 1)
        yb_oh_sl = torch.cat([yb_oh[:, 0:] + yb_oh[:, 2:].sum(1, keepdim=True), yb_oh[:, 1:2]], 1)
        
        
        # for the detection of small lesions we use the first two channels as background and the
        # others as foreground
        logs_ds_bg = logs[:, :2].max(1, keepdim=True)[0]
        logs_ds = torch.cat([logs_ds_bg, logs[:, 2:]], 1)
        yb_oh_ds = torch.cat([yb_oh[:, :2].sum(1, keepdim=True), yb_oh[:, 2:]], 1)
        
        loss_sl = self.ce_loss(logs_sl, yb_oh_sl, mask) + self.dice_loss(logs_sl, yb_oh_sl, mask)
        loss_ds = self.ce_loss_weighted(logs_ds, yb_oh_ds, mask) + self.dice_loss_weighted(logs_ds, yb_oh_ds, mask)
        
        return self.weight_sl * loss_sl + self.weight_ds * loss_ds
