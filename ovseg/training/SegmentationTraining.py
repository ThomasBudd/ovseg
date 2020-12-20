from ovseg.training.NetworkTraining import NetworkTraining
from ovseg.training.loss_functions import CE_dice_pyramid_loss


class SegmentationTraining(NetworkTraining):

    def initialise_loss(self):
        self.loss_fctn = CE_dice_pyramid_loss(**self.loss_params)

    def compute_batch_loss(self, batch):
        batch = batch[0].to(self.dev)
        if self.augmentation is not None:
            batch = self.augmentation.augment_batch(batch)
        xb, yb = batch[:, :-1], batch[:, -1:]
        out = self.network(xb)
        loss = self.loss_fctn(out, yb)
        return loss
