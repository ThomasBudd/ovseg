from ovseg.training.SegmentationTraining import SegmentationTraining
from ovseg.training.loss_functions import dice_pyramid_loss_class_ensembling, to_one_hot_encoding
import torch

class ClassEnsemblingTraining(SegmentationTraining):
    
    def __init__(self, *args,
                 prg_trn_sizes=None,
                 prg_trn_arch_params=None,
                 prg_trn_aug_params=None,
                 prg_trn_resize_on_the_fly=True,
                 **kwargs):
        if prg_trn_sizes is not None:
            raise NotImplementedError('Progressive traning is not implemented for class ensembling')
        super().__init__(*args, **kwargs)

    def initialise_loss(self):
        self.loss_fctn = dice_pyramid_loss_class_ensembling(**self.loss_params)

    def compute_batch_loss(self, batch):

        batch = batch.cuda()
        xb, yb = batch[:, :-1], batch[:, -1:]
        xb, yb = self.prg_trn_process_batch(xb, yb)

        if self.augmentation is not None:
            batch = torch.cat([xb, yb], 1)
            with torch.no_grad():
                batch = self.augmentation(batch)
            xb, yb = batch[:, :-1], batch[:, -1:]

        bin_pred = xb[:, -1:]
        yb = to_one_hot_encoding(yb, self.network.out_channels)
        out = self.network(xb)
        loss = self.loss_fctn(out, yb, bin_pred)
        return loss
