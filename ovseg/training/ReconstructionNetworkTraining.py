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

    def _prepare_data(self, data_tpl):
        xb, yb = data_tpl[0].to(self.dev), data_tpl[1].to(self.dev)
        return xb[0], yb[0]
