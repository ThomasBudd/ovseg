from ovseg.training.NetworkTraining import NetworkTraining
import torch


class RestaurationNetworkTraining(NetworkTraining):

    def initialise_loss(self):
        self.loss_fctn = torch.nn.MSELoss()

    def compute_batch_loss(self, batch):
        batch = batch.to(self.dev)
        xb, yb = batch[:, :1], batch[:, 1:]
        out = self.network(xb)
        loss = self.loss_fctn(out, yb)
        return loss
