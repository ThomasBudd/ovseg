from ovseg.training.NetworkTraining import NetworkTraining
from ovseg.training.loss_functions import CE_dice_pyramid_loss
import torch


class SegmentationTraining(NetworkTraining):

    def __init__(self, *args, image_key='image', label_key='label', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_key = image_key
        self.label_key = label_key

    def initialise_loss(self):
        self.loss_fctn = CE_dice_pyramid_loss(**self.loss_params)

    def compute_batch_loss(self, batch):
        xb, yb = batch[self.image_key].cuda(), batch[self.label_key].cuda()
        if self.augmentation is not None:
            b = torch.cat([xb, yb], 1)
            b = self.augmentation.augment_batch(b)
            xb, yb = b[:, :-1], b[:, -1:]
        out = self.network(xb)
        loss = self.loss_fctn(out, yb)
        return loss
