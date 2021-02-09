from ovseg.training.NetworkTraining import NetworkTraining
import torch


class ImageProcessingTraining(NetworkTraining):

    def __init__(self, *args, image_key='image', **kwargs):
        super().__init__(*args, **kwargs)
        self.image_key = image_key

    def initialise_loss(self):
        self.loss_fctn = torch.nn.MSELoss()

    def compute_batch_loss(self, batch):
        xb = batch[self.image_key].to(self.dev)
        out = self.network(xb)[0]
        loss = self.loss_fctn(out, xb)
        return loss
