from ovseg.training.NetworkTraining import NetworkTraining
import torch


class ReconstructionNetworkTraining(NetworkTraining):

    def initialise_loss(self):
        self.mse = torch.nn.MSELoss()
        self.l1loss = torch.nn.L1Loss()
        if hasattr(self.loss_params, 'l1weight'):
            self.weight_loss = self.loss_params['l1weight']
        else:
            print('loss_params does not have the key \'l1weight\' which '
                  'controls the balance between L2 and L1 loss. Initialise '
                  'as 0 (L2 loss).')
            self.weight_loss = 0
        if self.weight_loss < 0 or self.weight_loss > 1:
            raise ValueError('loss_params[\'l1weight\'] must be in [0, 1].')

    def loss_fctn(self, x1, x2):
        return (1 - self.weight_loss) * self.mse(x1, x2) + self.weight_loss * \
            self.l1loss(x1, x2)

    def compute_batch_loss(self, batch):
        xb, yb = batch[0][0].to(self.dev), batch[1][0].to(self.dev)
        out = self.network(xb)
        loss = self.loss_fctn(out, yb)
        return loss
