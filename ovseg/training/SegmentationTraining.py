from ovseg.training.NetworkTraining import NetworkTraining
from ovseg.training.loss_functions import CE_dice_pyramid_loss


class SegmentationTraining(NetworkTraining):

    def __init__(self, *args, n_im_channels: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_im_channels = n_im_channels

    def initialise_loss(self):
        self.loss_fctn = CE_dice_pyramid_loss(**self.loss_params)

    def compute_batch_loss(self, batch):

        batch = batch.cuda()
        if self.augmentation is not None:
            batch = self.augmentation(batch)

        xb, yb = batch[:, :self.n_im_channels], batch[:, self.n_im_channels:]
        out = self.network(xb)
        loss = self.loss_fctn(out, yb)
        return loss
